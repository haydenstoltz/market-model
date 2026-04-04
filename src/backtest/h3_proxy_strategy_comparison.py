from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class PeriodSpec:
    label: str
    start: str
    end: str | None


SUBPERIODS: tuple[PeriodSpec, ...] = (
    PeriodSpec(label="full_history", start="1900-01-01", end=None),
    PeriodSpec(label="1986_1999", start="1986-11-30", end="1999-12-31"),
    PeriodSpec(label="2000_2007", start="2000-01-01", end="2007-12-31"),
    PeriodSpec(label="2008_2012", start="2008-01-01", end="2012-12-31"),
    PeriodSpec(label="2013_2019", start="2013-01-01", end="2019-12-31"),
    PeriodSpec(label="2020_latest", start="2020-01-01", end=None),
)

PROXY_FEATURES: tuple[str, ...] = (
    "FEDFUNDS_lag1",
    "TB3MS_lag1",
    "GS10_lag1",
    "term_spread_lag1",
    "yc_10y_2y_lag1",
    "credit_spread_lag1",
)


def _perf_stats(returns: pd.Series) -> dict[str, float]:
    rets = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if rets.empty:
        return {"cagr": float("nan"), "vol": float("nan"), "sharpe": float("nan"), "max_dd": float("nan")}
    equity = (1.0 + rets).cumprod()
    years = len(rets) / 12.0
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(rets.std(ddof=0) * np.sqrt(12.0))
    sharpe = float((rets.mean() * 12.0) / vol) if vol > 0 else float("nan")
    drawdown = equity / equity.cummax() - 1.0
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": float(drawdown.min())}


def _build_strategy_from_signal(
    base: pd.DataFrame,
    signal_col: str,
) -> pd.DataFrame:
    out = base[["date", "bh_ret", "cash_ret"]].copy()
    signal = pd.to_numeric(base[signal_col], errors="coerce").fillna(0.0).astype(float)
    out["signal"] = signal
    out["weight"] = out["signal"].shift(1).fillna(0.0)

    prev_weight = out["weight"].shift(1).fillna(0.0)
    out["turnover"] = (out["weight"] - prev_weight).abs()
    out["cost"] = 0.0010 * out["turnover"]
    out["strat_ret_gross"] = out["weight"] * out["bh_ret"] + (1.0 - out["weight"]) * out["cash_ret"]
    out["strat_ret"] = out["strat_ret_gross"] - out["cost"]
    out["equity_curve_strat"] = (1.0 + out["strat_ret"]).cumprod()
    return out


def _strategy_summary(
    strategy_name: str,
    strat: pd.DataFrame,
) -> dict[str, float | int | str]:
    stats = _perf_stats(strat["strat_ret"])
    rets = pd.to_numeric(strat["strat_ret"], errors="coerce").dropna().astype(float)
    turnover = pd.to_numeric(strat["turnover"], errors="coerce")
    weight = pd.to_numeric(strat["weight"], errors="coerce")
    return {
        "strategy_type": strategy_name,
        "months": int(len(rets)),
        "final_equity": float((1.0 + rets).cumprod().iloc[-1]) if not rets.empty else float("nan"),
        "CAGR": stats["cagr"],
        "vol": stats["vol"],
        "Sharpe": stats["sharpe"],
        "max_drawdown": stats["max_dd"],
        "avg_turnover": float(turnover.mean()),
        "total_turnover": float(turnover.sum()),
        "hit_rate": float((rets > 0.0).mean()) if not rets.empty else float("nan"),
        "invested_fraction": float(weight.mean()),
    }


def _agreement_summary(
    sample: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
) -> dict[str, float | int | str]:
    y_true_i = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int).to_numpy()
    y_pred_i = pd.to_numeric(y_pred, errors="coerce").fillna(0).astype(int).to_numpy()
    cm = confusion_matrix(y_true_i, y_pred_i, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "strategy_type": model_name,
        "sample": sample,
        "months": int(y_true_i.shape[0]),
        "accuracy": float(accuracy_score(y_true_i, y_pred_i)),
        "precision": float(precision_score(y_true_i, y_pred_i, zero_division=0)),
        "recall": float(recall_score(y_true_i, y_pred_i, zero_division=0)),
        "f1": float(f1_score(y_true_i, y_pred_i, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _subperiod_mask(dates: pd.Series, spec: PeriodSpec) -> pd.Series:
    dt = pd.to_datetime(dates, errors="coerce")
    start_ts = pd.Timestamp(spec.start)
    end_ts = pd.Timestamp.max if spec.end is None else pd.Timestamp(spec.end)
    return (dt >= start_ts) & (dt <= end_ts)


def _majority_class(y: pd.Series, default: int = 0) -> int:
    y_i = pd.to_numeric(y, errors="coerce").dropna().astype(int)
    if y_i.empty:
        return int(default)
    counts = y_i.value_counts()
    if counts.empty:
        return int(default)
    return int(counts.index[0])


def build_proxy_vs_baseline(
    project_root: Path,
    tree_max_depth: int = 3,
) -> dict[str, Path]:
    outputs = project_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    state_path = outputs / "strategy_h3_macro_state_table.csv"
    if not state_path.exists():
        raise RuntimeError(
            f"Missing required state table: {state_path}. "
            "Run h3 macro-state diagnostics first."
        )

    state = pd.read_csv(state_path)
    state["date"] = pd.to_datetime(state["date"], errors="coerce")
    state = state.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    required_cols = ["signal_h3", "final_weight", "market_return", "cash_return", *PROXY_FEATURES]
    missing_cols = [c for c in required_cols if c not in state.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required columns in macro-state table: {missing_cols}")

    frame = state[["date", "signal_h3", "final_weight", "market_return", "cash_return", *PROXY_FEATURES]].copy()
    frame = frame.rename(columns={"market_return": "bh_ret", "cash_return": "cash_ret"})
    frame["signal_h3"] = pd.to_numeric(frame["signal_h3"], errors="coerce").fillna(0.0).astype(int)
    frame["final_weight"] = pd.to_numeric(frame["final_weight"], errors="coerce").fillna(0.0).astype(float)
    for feature in PROXY_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    frame = frame.dropna(subset=list(PROXY_FEATURES) + ["bh_ret", "cash_ret"]).reset_index(drop=True)

    X_all = frame[list(PROXY_FEATURES)].astype(float)
    y_signal = frame["signal_h3"].astype(int)

    n = len(frame)
    logit_signal = np.zeros(n, dtype=float)
    logit_proba = np.zeros(n, dtype=float)
    tree_signal = np.zeros(n, dtype=float)

    logit_coef_records: list[dict[str, float | str]] = []
    tree_imp_records: list[dict[str, float | str]] = []

    for i in range(n):
        X_train = X_all.iloc[:i]
        y_train = y_signal.iloc[:i]
        X_test = X_all.iloc[[i]]

        majority = _majority_class(y_train, default=0)

        if i == 0:
            logit_signal[i] = float(majority)
            logit_proba[i] = float(majority)
            tree_signal[i] = float(majority)
            continue

        if y_train.nunique() < 2:
            class_value = int(y_train.iloc[0]) if not y_train.empty else majority
            logit_signal[i] = float(class_value)
            logit_proba[i] = float(class_value)
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            logit = LogisticRegression(max_iter=5000, random_state=42)
            logit.fit(X_train_scaled, y_train.to_numpy(dtype=int))
            proba = float(logit.predict_proba(X_test_scaled)[0, 1])
            pred = 1.0 if proba >= 0.5 else 0.0
            logit_proba[i] = proba
            logit_signal[i] = pred

            for feature, coef in zip(PROXY_FEATURES, logit.coef_[0]):
                logit_coef_records.append(
                    {
                        "date": frame.loc[i, "date"].date().isoformat(),
                        "feature_name": feature,
                        "coef": float(coef),
                    }
                )

        tree = DecisionTreeClassifier(max_depth=int(tree_max_depth), random_state=42)
        tree.fit(X_train, y_train.to_numpy(dtype=int))
        tree_signal[i] = float(tree.predict(X_test)[0])
        for feature, imp in zip(PROXY_FEATURES, tree.feature_importances_):
            tree_imp_records.append(
                {
                    "date": frame.loc[i, "date"].date().isoformat(),
                    "feature_name": feature,
                    "importance": float(imp),
                }
            )

    frame["proxy_logit_proba_signal"] = logit_proba
    frame["proxy_logit_signal"] = logit_signal.astype(int)
    frame["proxy_tree_signal"] = tree_signal.astype(int)

    baseline = _build_strategy_from_signal(
        base=frame.assign(proxy_baseline_signal=frame["signal_h3"]),
        signal_col="proxy_baseline_signal",
    )
    baseline["strategy_type"] = "ridge_h3_baseline"
    baseline["signal_model"] = frame["signal_h3"].to_numpy(dtype=int)

    proxy_logit = _build_strategy_from_signal(base=frame, signal_col="proxy_logit_signal")
    proxy_logit["strategy_type"] = "proxy_logit"
    proxy_logit["signal_model"] = frame["proxy_logit_signal"].to_numpy(dtype=int)
    proxy_logit["signal_proba"] = frame["proxy_logit_proba_signal"].to_numpy(dtype=float)

    proxy_tree = _build_strategy_from_signal(base=frame, signal_col="proxy_tree_signal")
    proxy_tree["strategy_type"] = "proxy_tree"
    proxy_tree["signal_model"] = frame["proxy_tree_signal"].to_numpy(dtype=int)

    strategy_summary_rows = [
        _strategy_summary("ridge_h3_baseline", baseline),
        _strategy_summary("proxy_logit", proxy_logit),
        _strategy_summary("proxy_tree", proxy_tree),
    ]
    strategy_summary = pd.DataFrame(strategy_summary_rows)

    baseline_binary_weight = (pd.to_numeric(baseline["weight"], errors="coerce") > 0.0).astype(int)
    logit_binary_weight = (pd.to_numeric(proxy_logit["weight"], errors="coerce") > 0.0).astype(int)
    tree_binary_weight = (pd.to_numeric(proxy_tree["weight"], errors="coerce") > 0.0).astype(int)

    agreement_rows: list[dict[str, float | int | str]] = []
    for strategy_name, pred_binary in (("proxy_logit", logit_binary_weight), ("proxy_tree", tree_binary_weight)):
        for spec in SUBPERIODS:
            mask = _subperiod_mask(baseline["date"], spec)
            subset_true = baseline_binary_weight.loc[mask]
            subset_pred = pred_binary.loc[mask]
            if subset_true.empty:
                continue
            agreement_rows.append(
                _agreement_summary(
                    sample=spec.label,
                    y_true=subset_true,
                    y_pred=subset_pred,
                    model_name=strategy_name,
                )
            )

    agreement_df = pd.DataFrame(agreement_rows).sort_values(["strategy_type", "sample"]).reset_index(drop=True)

    predictions_out = frame[["date", "signal_h3", "proxy_logit_proba_signal", "proxy_logit_signal", "proxy_tree_signal"]].copy()
    predictions_out["baseline_weight"] = baseline["weight"].to_numpy(dtype=float)
    predictions_out["proxy_logit_weight"] = proxy_logit["weight"].to_numpy(dtype=float)
    predictions_out["proxy_tree_weight"] = proxy_tree["weight"].to_numpy(dtype=float)

    strategy_out = outputs / "strategy_h3_proxy_vs_baseline.csv"
    agreement_out = outputs / "strategy_h3_proxy_agreement.csv"
    pred_out = outputs / "strategy_h3_proxy_predictions.csv"
    logit_coef_out = outputs / "strategy_h3_proxy_logit_coef_timeseries.csv"
    tree_imp_out = outputs / "strategy_h3_proxy_tree_importance_timeseries.csv"
    logit_coef_summary_out = outputs / "strategy_h3_proxy_logit_coef_summary.csv"
    tree_imp_summary_out = outputs / "strategy_h3_proxy_tree_importance_summary.csv"

    strategy_summary.to_csv(strategy_out, index=False)
    agreement_df.to_csv(agreement_out, index=False)
    predictions_out.to_csv(pred_out, index=False)

    coef_df = pd.DataFrame(logit_coef_records)
    imp_df = pd.DataFrame(tree_imp_records)
    coef_df.to_csv(logit_coef_out, index=False)
    imp_df.to_csv(tree_imp_out, index=False)

    if not coef_df.empty:
        coef_summary = (
            coef_df.groupby("feature_name")["coef"]
            .agg(mean_coef="mean", mean_abs_coef=lambda s: float(np.mean(np.abs(s.to_numpy(dtype=float)))))
            .reset_index()
            .sort_values("mean_abs_coef", ascending=False)
            .reset_index(drop=True)
        )
    else:
        coef_summary = pd.DataFrame(columns=["feature_name", "mean_coef", "mean_abs_coef"])

    if not imp_df.empty:
        imp_summary = (
            imp_df.groupby("feature_name")["importance"]
            .mean()
            .reset_index(name="mean_importance")
            .sort_values("mean_importance", ascending=False)
            .reset_index(drop=True)
        )
    else:
        imp_summary = pd.DataFrame(columns=["feature_name", "mean_importance"])

    coef_summary.to_csv(logit_coef_summary_out, index=False)
    imp_summary.to_csv(tree_imp_summary_out, index=False)

    return {
        "strategy_summary": strategy_out,
        "agreement_summary": agreement_out,
        "predictions": pred_out,
        "logit_coef_ts": logit_coef_out,
        "tree_importance_ts": tree_imp_out,
        "logit_coef_summary": logit_coef_summary_out,
        "tree_importance_summary": tree_imp_summary_out,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone h3 macro-proxy strategy comparison vs ridge baseline.")
    parser.add_argument("--project-root", type=str, default=".", help="Project root path.")
    parser.add_argument("--tree-max-depth", type=int, default=3, help="Proxy tree depth.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    artifacts = build_proxy_vs_baseline(
        project_root=project_root,
        tree_max_depth=int(args.tree_max_depth),
    )
    print("[h3-proxy-vs-baseline] wrote:")
    for key, path in artifacts.items():
        print(f" - {key}: {path}")


if __name__ == "__main__":
    main()
