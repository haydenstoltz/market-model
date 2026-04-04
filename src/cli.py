from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest.walkforward import run_walkforward_backtest
from src.data.fred import load_macro_monthly
from src.data.market import build_monthly_dataset
from src.features.build import FEATURE_BATCH_1_COLUMNS, FEATURE_BATCH_2_COLUMNS, build_features
from src.targets.build import build_targets


REGIME_GATE_CHOICES = ("none", "fedfunds_high", "inflation_low", "both")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _perf_stats(returns: pd.Series) -> dict[str, float]:
    rets = returns.dropna().astype(float)
    if rets.empty:
        return {"cagr": float("nan"), "vol": float("nan"), "sharpe": float("nan"), "max_dd": float("nan")}

    equity = (1.0 + rets).cumprod()
    n = len(rets)
    years = n / 12.0
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(rets.std(ddof=0) * np.sqrt(12.0))
    sharpe = float((rets.mean() * 12.0) / vol) if vol > 0 else float("nan")
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": max_dd}


def _build_strategy_from_signal(
    aligned: pd.DataFrame,
    signal: pd.Series,
    gate_mask: pd.Series | None = None,
) -> pd.DataFrame:
    strat = pd.DataFrame(index=aligned.index)
    strat["signal"] = signal.astype(float)
    strat["weight"] = strat["signal"].shift(1).fillna(0.0)

    if gate_mask is None:
        gate_series = pd.Series(True, index=aligned.index, dtype=bool)
    else:
        gate_series = pd.Series(gate_mask, index=aligned.index).fillna(False).astype(bool)
    strat["gate_active"] = gate_series.astype(float)
    strat["weight"] = strat["weight"].where(gate_series, 0.0)

    strat["cash_ret"] = aligned["cash_ret"]

    prev_weight = strat["weight"].shift(1).fillna(0.0)
    strat["turnover"] = (strat["weight"] - prev_weight).abs()
    strat["cost"] = 0.0010 * strat["turnover"]  # 10 bps cost when allocation flips

    strat["strat_ret_gross"] = strat["weight"] * aligned["return_1m"] + (1.0 - strat["weight"]) * strat["cash_ret"]
    strat["strat_ret"] = strat["strat_ret_gross"] - strat["cost"]
    strat["bh_ret"] = aligned["return_1m"]

    strat["equity_curve_strat"] = (1.0 + strat["strat_ret"]).cumprod()
    strat["equity_curve_bh"] = (1.0 + strat["bh_ret"]).cumprod()

    return strat.reset_index().rename(columns={"index": "date"})


def _build_long_only_strategy(
    aligned: pd.DataFrame,
    threshold: float,
    gate_mask: pd.Series | None = None,
) -> pd.DataFrame:
    signal = (aligned["y_pred"] > threshold).astype(float)
    return _build_strategy_from_signal(aligned=aligned, signal=signal, gate_mask=gate_mask)


def _build_tiered_strategy(aligned: pd.DataFrame, tier_threshold: float) -> pd.DataFrame:
    signal = np.where(
        aligned["y_pred"] <= 0.0,
        0.0,
        np.where(aligned["y_pred"] <= tier_threshold, 0.5, 1.0),
    )
    signal_series = pd.Series(signal, index=aligned.index, dtype=float)
    return _build_strategy_from_signal(aligned=aligned, signal=signal_series)


def _build_run_summary(
    strat: pd.DataFrame,
    model_name: str,
    horizon: int,
    strategy_name: str,
    strategy_type: str,
    threshold: float,
) -> dict[str, float | int | str]:
    strat_rets = strat["strat_ret"].astype(float)
    bh_rets = strat["bh_ret"].astype(float)

    strat_stats = _perf_stats(strat_rets)
    bh_stats = _perf_stats(bh_rets)

    start_date = pd.to_datetime(strat["date"]).min().date().isoformat() if not strat.empty else ""
    end_date = pd.to_datetime(strat["date"]).max().date().isoformat() if not strat.empty else ""

    final_equity_strat = float(strat["equity_curve_strat"].iloc[-1]) if not strat.empty else float("nan")
    final_equity_bh = float(strat["equity_curve_bh"].iloc[-1]) if not strat.empty else float("nan")

    hit_rate_strat = float((strat_rets > 0.0).mean()) if not strat.empty else float("nan")
    hit_rate_bh = float((bh_rets > 0.0).mean()) if not strat.empty else float("nan")

    avg_turnover = float(strat["turnover"].mean()) if not strat.empty else float("nan")
    total_turnover = float(strat["turnover"].sum()) if not strat.empty else float("nan")
    invested_fraction = float(strat["weight"].mean()) if not strat.empty else float("nan")

    return {
        "strategy": strategy_name,
        "strategy_type": strategy_type,
        "threshold": float(threshold),
        "model": model_name,
        "horizon": int(horizon),
        "start_date": start_date,
        "end_date": end_date,
        "months": int(len(strat)),
        "final_equity_strat": final_equity_strat,
        "final_equity_bh": final_equity_bh,
        "CAGR_strat": strat_stats["cagr"],
        "CAGR_bh": bh_stats["cagr"],
        "vol_strat": strat_stats["vol"],
        "vol_bh": bh_stats["vol"],
        "Sharpe_strat": strat_stats["sharpe"],
        "Sharpe_bh": bh_stats["sharpe"],
        "max_drawdown_strat": strat_stats["max_dd"],
        "max_drawdown_bh": bh_stats["max_dd"],
        "hit_rate_strat": hit_rate_strat,
        "hit_rate_bh": hit_rate_bh,
        "avg_turnover": avg_turnover,
        "total_turnover": total_turnover,
        "invested_fraction": invested_fraction,
    }


def _build_strategy_metric_summary(strat: pd.DataFrame) -> dict[str, float | int]:
    strat_rets = (
        pd.to_numeric(strat["strat_ret"], errors="coerce").dropna()
        if "strat_ret" in strat.columns
        else pd.Series(dtype=float)
    )
    turnover = (
        pd.to_numeric(strat["turnover"], errors="coerce")
        if "turnover" in strat.columns
        else pd.Series(dtype=float)
    )
    weight = (
        pd.to_numeric(strat["weight"], errors="coerce")
        if "weight" in strat.columns
        else pd.Series(dtype=float)
    )

    perf = _perf_stats(strat_rets)
    final_equity = float((1.0 + strat_rets).cumprod().iloc[-1]) if not strat_rets.empty else float("nan")
    hit_rate = float((strat_rets > 0.0).mean()) if not strat_rets.empty else float("nan")

    return {
        "months": int(len(strat_rets)),
        "final_equity": final_equity,
        "CAGR": perf["cagr"],
        "vol": perf["vol"],
        "Sharpe": perf["sharpe"],
        "max_drawdown": perf["max_dd"],
        "avg_turnover": float(turnover.mean()) if not turnover.empty else float("nan"),
        "total_turnover": float(turnover.sum()) if not turnover.empty else float("nan"),
        "hit_rate": hit_rate,
        "invested_fraction": float(weight.mean()) if not weight.empty else float("nan"),
    }


def _build_h3_smooth_signal(y_pred_h3: pd.Series, scale: float) -> pd.Series:
    scale_value = float(scale)
    if scale_value <= 0.0:
        raise ValueError("--h3-smooth-scale must be a positive float.")

    pred = pd.to_numeric(y_pred_h3, errors="coerce")
    weight_raw = pred / scale_value
    weight_clipped = weight_raw.clip(lower=0.0, upper=1.0).fillna(0.0)
    return weight_clipped.astype(float)


def _apply_weight_floor_overlay(strat: pd.DataFrame, floor_weight: float) -> pd.DataFrame:
    floor_value = float(floor_weight)
    if floor_value < 0.0 or floor_value > 1.0:
        raise ValueError("--h3-floor-weight must be within [0, 1].")

    out = strat.copy()
    baseline_weight = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    out["baseline_weight"] = baseline_weight
    out["weight"] = baseline_weight.clip(lower=floor_value, upper=1.0)

    prev_weight = out["weight"].shift(1).fillna(0.0)
    out["turnover"] = (out["weight"] - prev_weight).abs()
    out["cost"] = 0.0010 * out["turnover"]
    out["strat_ret_gross"] = out["weight"] * out["bh_ret"] + (1.0 - out["weight"]) * out["cash_ret"]
    out["strat_ret"] = out["strat_ret_gross"] - out["cost"]
    out["equity_curve_strat"] = (1.0 + out["strat_ret"]).cumprod()
    out["equity_curve_bh"] = (1.0 + out["bh_ret"]).cumprod()
    return out


def _build_h3_regime_gate_mask(
    aligned_h3: pd.DataFrame,
    regime_gate: str,
) -> tuple[pd.Series | None, float, float]:
    gate_name = str(regime_gate).strip().lower()
    if gate_name not in REGIME_GATE_CHOICES:
        raise ValueError(f"Invalid --regime-gate '{regime_gate}'. Expected one of: {list(REGIME_GATE_CHOICES)}")

    fed = pd.to_numeric(aligned_h3["FEDFUNDS_lag1"], errors="coerce")
    inflation = pd.to_numeric(aligned_h3["inflation_yoy_lag1"], errors="coerce")

    fed_median = float(fed.dropna().median()) if fed.notna().any() else float("nan")
    inflation_median = float(inflation.dropna().median()) if inflation.notna().any() else float("nan")

    if gate_name == "none":
        return None, fed_median, inflation_median

    if np.isnan(fed_median) or np.isnan(inflation_median):
        raise RuntimeError("Cannot compute regime gate medians for h3 (insufficient FEDFUNDS_lag1/inflation_yoy_lag1 values).")

    fed_high = fed > fed_median
    inflation_low = inflation <= inflation_median

    if gate_name == "fedfunds_high":
        gate_mask = fed_high
    elif gate_name == "inflation_low":
        gate_mask = inflation_low
    elif gate_name == "both":
        gate_mask = fed_high & inflation_low
    else:
        raise ValueError(f"Invalid --regime-gate '{regime_gate}'. Expected one of: {list(REGIME_GATE_CHOICES)}")

    return gate_mask.fillna(False).astype(bool), fed_median, inflation_median


def _parse_threshold_grid(threshold_grid: str) -> list[float]:
    thresholds: list[float] = []
    for raw_value in threshold_grid.split(","):
        value = raw_value.strip()
        if not value:
            continue
        try:
            thresholds.append(float(value))
        except ValueError as exc:
            raise ValueError(f"Invalid threshold value '{value}' in --threshold-grid.") from exc

    if not thresholds:
        raise ValueError("--threshold-grid did not contain any numeric threshold values.")

    return thresholds


def _summarize_ridge_coefficients(
    coefficient_records: pd.DataFrame,
    horizon: int = 1,
    nontrivial_threshold: float = 1e-6,
) -> pd.DataFrame:
    columns = ["feature_name", "mean_abs_coef", "mean_coef", "std_coef", "fraction_nontrivial_coef"]
    if coefficient_records.empty:
        return pd.DataFrame(columns=columns)

    coef_df = coefficient_records.copy()
    if "horizon" in coef_df.columns:
        coef_df = coef_df[coef_df["horizon"] == int(horizon)]
    if coef_df.empty:
        return pd.DataFrame(columns=columns)

    grouped = coef_df.groupby("feature_name")["coef"]
    summary = grouped.agg(
        mean_abs_coef=lambda s: float(np.mean(np.abs(s.to_numpy(dtype=float)))),
        mean_coef=lambda s: float(np.mean(s.to_numpy(dtype=float))),
        std_coef=lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0)),
        fraction_nontrivial_coef=lambda s: float((np.abs(s.to_numpy(dtype=float)) > float(nontrivial_threshold)).mean()),
    )
    summary = summary.reset_index().sort_values("mean_abs_coef", ascending=False).reset_index(drop=True)
    return summary[columns]


def _write_ridge_coefficient_outputs(
    coefficient_records: pd.DataFrame,
    suffix: str = "",
    summary_horizon: int = 1,
) -> pd.DataFrame:
    if coefficient_records.empty:
        print("[backtest] ridge coefficient diagnostics unavailable (no coefficients recorded)")
        return _summarize_ridge_coefficients(coefficient_records, horizon=summary_horizon)

    windows_out = ensure_parent(f"outputs/ridge_coef_windows{suffix}.csv")
    coefficient_records.to_csv(windows_out, index=False)
    print(f"[backtest] wrote ridge coefficient windows -> {windows_out}")

    summary_df = _summarize_ridge_coefficients(coefficient_records, horizon=summary_horizon)
    summary_out = ensure_parent(f"outputs/ridge_coef_summary{suffix}.csv")
    summary_df.to_csv(summary_out, index=False)
    print(f"[backtest] wrote ridge coefficient summary -> {summary_out}")
    return summary_df


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional at runtime
        print(f"[backtest] chart generation skipped ({exc})")
        return None
    return plt


def _strategy_plot_label(strategy_name: str, strategy_type: str, threshold: float) -> str:
    if strategy_name == "baseline_binary":
        return "binary (threshold=0)"
    if strategy_name == "thresholded":
        return f"binary (threshold={float(threshold):0.4f})"
    if strategy_type == "tiered":
        return f"tiered (threshold={float(threshold):0.4f})"
    return strategy_name


def _write_strategy_charts(
    summary_df: pd.DataFrame,
    strategy_frames: dict[str, pd.DataFrame],
    model_name: str,
) -> None:
    if summary_df.empty:
        print("[backtest] strategy chart generation skipped (empty summary)")
        return

    plt = _load_pyplot()
    if plt is None:
        return

    plot_df = summary_df.copy()
    plot_df["label"] = plot_df.apply(
        lambda r: _strategy_plot_label(
            strategy_name=str(r["strategy"]),
            strategy_type=str(r["strategy_type"]),
            threshold=float(r["threshold"]),
        ),
        axis=1,
    )
    plot_df["Sharpe_strat"] = pd.to_numeric(plot_df["Sharpe_strat"], errors="coerce")
    plot_df["final_equity_strat"] = pd.to_numeric(plot_df["final_equity_strat"], errors="coerce")

    sharpe_non_null = plot_df["Sharpe_strat"].dropna()
    best_idx = sharpe_non_null.idxmax() if not sharpe_non_null.empty else plot_df.index[0]
    best_row = plot_df.loc[best_idx]
    best_strategy = str(best_row["strategy"])

    colors = ["tab:green" if idx == best_idx else "tab:blue" for idx in plot_df.index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(plot_df["label"], plot_df["Sharpe_strat"], color=colors)
    axes[0].set_title("Sharpe by Strategy")
    axes[0].set_ylabel("Sharpe")
    axes[0].tick_params(axis="x", rotation=18)

    axes[1].bar(plot_df["label"], plot_df["final_equity_strat"], color=colors)
    axes[1].set_title("Final Equity by Strategy")
    axes[1].set_ylabel("Final Equity")
    axes[1].tick_params(axis="x", rotation=18)

    fig.suptitle(
        "Most Effective Method (by Sharpe): "
        f"{best_row['label']} | model={model_name} | Sharpe={float(best_row['Sharpe_strat']):0.4f}"
    )
    fig.tight_layout()
    effectiveness_out = ensure_parent("outputs/strategy_method_effectiveness.png")
    fig.savefig(effectiveness_out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[backtest] wrote strategy chart -> {effectiveness_out}")

    label_by_strategy = {
        str(row["strategy"]): str(row["label"])
        for _, row in plot_df.iterrows()
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    for strategy_name, strat in strategy_frames.items():
        if strat.empty:
            continue
        dates = pd.to_datetime(strat["date"])
        label = label_by_strategy.get(strategy_name, strategy_name)
        line_width = 2.6 if strategy_name == best_strategy else 1.6
        if strategy_name == best_strategy:
            label = f"{label} [best Sharpe]"
        ax.plot(dates, strat["equity_curve_strat"], label=label, linewidth=line_width)

    baseline = strategy_frames.get("baseline_binary")
    if baseline is not None and not baseline.empty:
        ax.plot(
            pd.to_datetime(baseline["date"]),
            baseline["equity_curve_bh"],
            label="buy & hold",
            linewidth=1.8,
            linestyle="--",
            color="black",
        )

    ax.set_title(f"Strategy Returns Comparison (model={model_name})")
    ax.set_ylabel("Equity Curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    returns_out = ensure_parent("outputs/strategy_returns_comparison.png")
    fig.savefig(returns_out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[backtest] wrote returns chart -> {returns_out}")


def _write_residual_chart(
    features: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> None:
    if "residual_method" not in features.columns:
        return

    plt = _load_pyplot()
    if plt is None:
        return

    residual = features["residual_method"].copy()
    if start:
        residual = residual[residual.index >= pd.to_datetime(start)]
    if end:
        residual = residual[residual.index <= pd.to_datetime(end)]
    residual = residual.dropna()
    if residual.empty:
        print("[backtest] residual chart skipped (no valid residual values in selected window)")
        return

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(pd.to_datetime(residual.index), residual.astype(float), color="tab:orange", linewidth=1.4)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("residual_method Over Time")
    ax.set_ylabel("Residual")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    residual_out = ensure_parent("outputs/residual_method_over_time.png")
    fig.savefig(residual_out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[backtest] wrote residual chart -> {residual_out}")


def run_data(cfg: dict) -> None:
    market_df = build_monthly_dataset(
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        frequency=cfg.get("frequency", "ME"),
        seed=int(cfg.get("seed", 42)),
        market_source=cfg.get("market_source", "yahoo"),
        market_ticker=cfg.get("market_ticker", "^GSPC"),
    )
    if market_df.empty:
        raise RuntimeError("Market dataframe is empty after SP500 ingestion.")
    print(
        "[data] market span "
        f"{market_df.index.min().date()} -> {market_df.index.max().date()} "
        f"({len(market_df)} rows)"
    )

    macro_df = load_macro_monthly(
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        series_ids=cfg.get("macro_series_ids", []),
        frequency=cfg.get("frequency", "ME"),
    )

    if {"GS10", "TB3MS"}.issubset(set(macro_df.columns)):
        macro_df["term_spread"] = macro_df["GS10"] - macro_df["TB3MS"]
    if "UNRATE" in macro_df.columns:
        macro_df["unemployment_rate"] = macro_df["UNRATE"]
    if "CPIAUCSL" in macro_df.columns:
        macro_df["inflation_mom"] = macro_df["CPIAUCSL"].pct_change()

    df = market_df.join(macro_df, how="inner")
    if df.shape[1] <= 3:
        raise RuntimeError(
            "Joined raw dataset has <=3 columns; expected market plus macro columns."
        )

    out_path = ensure_parent(cfg["paths"]["raw_data"])
    df.to_csv(out_path)
    print(f"[data] wrote {len(df)} rows x {df.shape[1]} cols -> {out_path}")


def run_features(
    cfg: dict,
    use_residual_method: bool = False,
    residual_warmup: int = 60,
    use_feature_batch_1: bool = False,
    use_feature_batch_2: bool = False,
) -> None:
    raw_path = Path(cfg["paths"]["raw_data"])
    df = pd.read_csv(raw_path, parse_dates=["date"], index_col="date")

    feats = build_features(
        df=df,
        ma_window=int(cfg["ma_window"]),
        use_residual_method=use_residual_method,
        residual_warmup=int(residual_warmup),
        use_feature_batch_1=use_feature_batch_1,
        use_feature_batch_2=use_feature_batch_2,
    )
    out_path = ensure_parent(cfg["paths"]["features"])
    feats.to_csv(out_path)
    print(f"[features] wrote {len(feats)} rows -> {out_path}")


def run_targets(cfg: dict) -> None:
    raw_path = Path(cfg["paths"]["raw_data"])
    df = pd.read_csv(raw_path, parse_dates=["date"], index_col="date")

    targs = build_targets(
        df=df,
        horizons=list(cfg["horizons"]),
        ma_window=int(cfg["ma_window"]),
    )
    non_null = targs.notna().sum().to_dict()
    print(f"[targets] non-null counts {non_null}")

    out_path = ensure_parent(cfg["paths"]["targets"])
    targs.to_csv(out_path)
    print(f"[targets] wrote {len(targs)} rows -> {out_path}")


def run_backtest(
    cfg: dict,
    model_name: str = "ridge",
    start: str | None = None,
    end: str | None = None,
    signal_threshold: float = 0.0,
    tier_threshold: float = 0.0025,
    threshold_grid: str | None = None,
    use_residual_method: bool = False,
    residual_warmup: int = 60,
    use_feature_batch_1: bool = False,
    use_feature_batch_2: bool = False,
    top_k_features: int | None = None,
    regime_gate: str = "none",
    h3_confirm_with_h1: bool = False,
    h3_smooth_scale: float | None = None,
    h3_floor_weight: float | None = None,
) -> None:
    feature_path = Path(cfg["paths"]["features"])
    target_path = Path(cfg["paths"]["targets"])
    raw_path = Path(cfg["paths"]["raw_data"])

    targets = pd.read_csv(target_path, parse_dates=["date"], index_col="date")
    raw = pd.read_csv(raw_path, parse_dates=["date"], index_col="date")

    print(f"[backtest] feature_batch_1 enabled={bool(use_feature_batch_1)}")
    print(f"[backtest] feature_batch_2 enabled={bool(use_feature_batch_2)}")
    regime_gate = str(regime_gate).strip().lower()
    if regime_gate not in REGIME_GATE_CHOICES:
        raise ValueError(f"Invalid --regime-gate '{regime_gate}'. Expected one of: {list(REGIME_GATE_CHOICES)}")
    print(f"[backtest] regime_gate={regime_gate}")
    print(f"[backtest] h3_confirm_with_h1={bool(h3_confirm_with_h1)}")
    if h3_smooth_scale is not None:
        h3_smooth_scale = float(h3_smooth_scale)
        if h3_smooth_scale <= 0.0:
            raise ValueError("--h3-smooth-scale must be a positive float.")
    print(f"[backtest] h3_smooth_scale={h3_smooth_scale}")
    if h3_floor_weight is not None:
        h3_floor_weight = float(h3_floor_weight)
        if h3_floor_weight < 0.0 or h3_floor_weight > 1.0:
            raise ValueError("--h3-floor-weight must be within [0, 1].")
    print(f"[backtest] h3_floor_weight={h3_floor_weight}")
    if use_residual_method:
        print(f"[backtest] residual_method enabled=True warmup={int(residual_warmup)}")
    else:
        print("[backtest] residual_method enabled=False")

    if top_k_features is not None:
        if int(top_k_features) <= 0:
            raise ValueError("--top-k-features must be a positive integer.")
        if model_name.lower() != "ridge":
            raise ValueError("--top-k-features is currently supported only for --model ridge.")
        if use_residual_method or use_feature_batch_1 or use_feature_batch_2:
            raise ValueError(
                "--top-k-features currently supports the baseline feature set only; disable optional feature flags."
            )

    if use_residual_method or use_feature_batch_1 or use_feature_batch_2:
        features = build_features(
            df=raw.copy(),
            ma_window=int(cfg["ma_window"]),
            use_residual_method=bool(use_residual_method),
            residual_warmup=int(residual_warmup),
            use_feature_batch_1=bool(use_feature_batch_1),
            use_feature_batch_2=bool(use_feature_batch_2),
        )
    else:
        features = pd.read_csv(feature_path, parse_dates=["date"], index_col="date")

    has_residual = "residual_method" in features.columns
    print(f"[backtest] residual_method in feature columns: {has_residual}")
    if has_residual:
        first_valid = features["residual_method"].first_valid_index()
        if first_valid is None:
            print("[backtest] residual_method first valid feature date: none")
        else:
            print(f"[backtest] residual_method first valid feature date: {first_valid.date()}")

    if use_feature_batch_1:
        feature_diag = features
        if start:
            feature_diag = feature_diag[feature_diag.index >= pd.to_datetime(start)]
        if end:
            feature_diag = feature_diag[feature_diag.index <= pd.to_datetime(end)]

        for feature_name in FEATURE_BATCH_1_COLUMNS:
            included = feature_name in feature_diag.columns
            has_non_null = bool(feature_diag[feature_name].notna().any()) if included else False
            first_valid = feature_diag[feature_name].first_valid_index() if included else None
            first_valid_str = "none" if first_valid is None else first_valid.date().isoformat()
            print(
                f"[backtest] feature_batch_1 {feature_name} "
                f"included={included} non_null={has_non_null} first_valid={first_valid_str}"
            )

    if use_feature_batch_2:
        feature_diag = features
        if start:
            feature_diag = feature_diag[feature_diag.index >= pd.to_datetime(start)]
        if end:
            feature_diag = feature_diag[feature_diag.index <= pd.to_datetime(end)]

        for feature_name in FEATURE_BATCH_2_COLUMNS:
            included = feature_name in feature_diag.columns
            has_non_null = bool(feature_diag[feature_name].notna().any()) if included else False
            first_valid = feature_diag[feature_name].first_valid_index() if included else None
            first_valid_str = "none" if first_valid is None else first_valid.date().isoformat()
            print(
                f"[backtest] feature_batch_2 {feature_name} "
                f"included={included} non_null={has_non_null} first_valid={first_valid_str}"
            )

    selected_feature_names = list(features.columns)

    regime_columns = ["term_spread_lag1", "FEDFUNDS_lag1", "inflation_yoy_lag1"]
    h3_gate_columns = ["FEDFUNDS_lag1", "inflation_yoy_lag1"]
    available_regime_columns = [col for col in regime_columns if col in features.columns]
    missing_regime_columns = [col for col in regime_columns if col not in features.columns]

    if top_k_features is not None:
        ranking_result = run_walkforward_backtest(
            features=features,
            targets=targets,
            horizons=list(cfg["horizons"]),
            train_min_periods=int(cfg.get("train_min_periods", 36)),
            model_name=model_name,
            start=start,
            end=end,
            log_window_runtime=False,
        )
        ranking_summary = _write_ridge_coefficient_outputs(
            coefficient_records=ranking_result.coefficient_records,
            suffix=f"_ranking_source_top{int(top_k_features)}",
            summary_horizon=1,
        )
        if ranking_summary.empty:
            raise RuntimeError("Could not rank features for --top-k-features: no ridge coefficients were recorded.")

        selected_feature_names = ranking_summary["feature_name"].head(int(top_k_features)).tolist()
        if not selected_feature_names:
            raise RuntimeError("Feature ranking returned no features for --top-k-features.")

        print(
            f"[backtest] top_k_features={int(top_k_features)} selected_count={len(selected_feature_names)} "
            f"selected_features={selected_feature_names}"
        )
        features_for_run = features[selected_feature_names].copy()
    else:
        features_for_run = features

    result = run_walkforward_backtest(
        features=features_for_run,
        targets=targets,
        horizons=list(cfg["horizons"]),
        train_min_periods=int(cfg.get("train_min_periods", 36)),
        model_name=model_name,
        start=start,
        end=end,
    )

    if model_name.lower() == "ridge":
        coef_suffix = "" if top_k_features is None else f"_top{int(top_k_features)}"
        coef_summary = _write_ridge_coefficient_outputs(
            coefficient_records=result.coefficient_records,
            suffix=coef_suffix,
            summary_horizon=1,
        )
        if not coef_summary.empty:
            print("[backtest] ridge top features by mean_abs_coef")
            print(coef_summary.head(15).to_string(index=False, float_format=lambda x: f"{x:0.8f}"))

    out_path = ensure_parent(cfg["paths"]["predictions"])
    result.predictions.to_csv(out_path, index=False)

    print(f"[backtest] wrote predictions -> {out_path}")
    print("[backtest] metrics")
    print(result.metrics.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))

    h1 = result.predictions[result.predictions["horizon"] == 1].copy()
    if h1.empty:
        raise RuntimeError("No horizon=1 predictions found; cannot compute strategy metrics.")

    h1["date"] = pd.to_datetime(h1["date"])
    h1 = h1.set_index("date").sort_index()

    aligned = h1.join(raw[["return_1m", "TB3MS"]], how="inner")
    if available_regime_columns:
        aligned = aligned.join(features[available_regime_columns], how="left")
    aligned["cash_ret"] = (aligned["TB3MS"].shift(1) / 100.0) / 12.0
    aligned = aligned.dropna(subset=["return_1m", "cash_ret"])
    if aligned.empty:
        raise RuntimeError("No overlap between horizon=1 predictions and market return_1m.")

    non_zero_mask = (aligned["y_pred"] != 0.0) & (aligned["y_true"] != 0.0)
    if non_zero_mask.any():
        sign_hit_rate = float(
            (np.sign(aligned.loc[non_zero_mask, "y_pred"]) == np.sign(aligned.loc[non_zero_mask, "y_true"])).mean()
        )
    else:
        sign_hit_rate = float("nan")

    baseline_threshold = 0.0
    baseline_strat = _build_long_only_strategy(aligned=aligned, threshold=baseline_threshold)

    strat_out = ensure_parent("outputs/strategy_h1.csv")
    baseline_strat.to_csv(strat_out, index=False)

    baseline_stats = _perf_stats(baseline_strat["strat_ret"])
    bh_stats = _perf_stats(baseline_strat["bh_ret"])

    print("[backtest] strategy_h1")
    print(f" sign_hit_rate_nonzero: {sign_hit_rate:0.6f}")
    print(
        " baseline strategy: "
        f"CAGR={baseline_stats['cagr']:0.6f}, Vol={baseline_stats['vol']:0.6f}, "
        f"Sharpe={baseline_stats['sharpe']:0.6f}, MaxDD={baseline_stats['max_dd']:0.6f}, "
        f"AvgTurnover={baseline_strat['turnover'].mean():0.6f}, AvgCost={baseline_strat['cost'].mean():0.6f}"
    )
    print(
        " buy_hold: "
        f"CAGR={bh_stats['cagr']:0.6f}, Vol={bh_stats['vol']:0.6f}, "
        f"Sharpe={bh_stats['sharpe']:0.6f}, MaxDD={bh_stats['max_dd']:0.6f}"
    )
    print(f"[backtest] wrote strategy -> {strat_out}")

    threshold_to_use = float(signal_threshold)
    if np.isclose(threshold_to_use, 0.0):
        threshold_to_use = 0.01

    threshold_strat = _build_long_only_strategy(aligned=aligned, threshold=threshold_to_use)
    threshold_stats = _perf_stats(threshold_strat["strat_ret"])

    threshold_out = ensure_parent("outputs/strategy_h1_threshold.csv")
    threshold_strat.to_csv(threshold_out, index=False)

    print("[backtest] strategy_h1_threshold")
    print(f" threshold: {threshold_to_use:0.6f}")
    print(
        " thresholded strategy: "
        f"CAGR={threshold_stats['cagr']:0.6f}, Vol={threshold_stats['vol']:0.6f}, "
        f"Sharpe={threshold_stats['sharpe']:0.6f}, MaxDD={threshold_stats['max_dd']:0.6f}, "
        f"AvgTurnover={threshold_strat['turnover'].mean():0.6f}, AvgCost={threshold_strat['cost'].mean():0.6f}"
    )
    print(f"[backtest] wrote threshold strategy -> {threshold_out}")

    tiered_strat = _build_tiered_strategy(aligned=aligned, tier_threshold=float(tier_threshold))
    tiered_stats = _perf_stats(tiered_strat["strat_ret"])

    tiered_out = ensure_parent("outputs/strategy_h1_tiered.csv")
    tiered_strat.to_csv(tiered_out, index=False)

    print("[backtest] strategy_h1_tiered")
    print(f" tier_threshold: {float(tier_threshold):0.6f}")
    print(
        " tiered strategy: "
        f"CAGR={tiered_stats['cagr']:0.6f}, Vol={tiered_stats['vol']:0.6f}, "
        f"Sharpe={tiered_stats['sharpe']:0.6f}, MaxDD={tiered_stats['max_dd']:0.6f}, "
        f"AvgTurnover={tiered_strat['turnover'].mean():0.6f}, AvgCost={tiered_strat['cost'].mean():0.6f}"
    )
    print(f"[backtest] wrote tiered strategy -> {tiered_out}")

    summary_rows = [
        _build_run_summary(
            strat=baseline_strat,
            model_name=model_name,
            horizon=1,
            strategy_name="baseline_binary",
            strategy_type="binary",
            threshold=baseline_threshold,
        ),
        _build_run_summary(
            strat=threshold_strat,
            model_name=model_name,
            horizon=1,
            strategy_name="thresholded",
            strategy_type="binary",
            threshold=threshold_to_use,
        ),
        _build_run_summary(
            strat=tiered_strat,
            model_name=model_name,
            horizon=1,
            strategy_name="tiered",
            strategy_type="tiered",
            threshold=float(tier_threshold),
        ),
    ]

    summary_out = ensure_parent("outputs/strategy_run_summary.csv")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_out, index=False)
    print(f"[backtest] wrote strategy summary -> {summary_out}")

    _write_strategy_charts(
        summary_df=summary_df,
        strategy_frames={
            "baseline_binary": baseline_strat,
            "thresholded": threshold_strat,
            "tiered": tiered_strat,
        },
        model_name=model_name,
    )
    _write_residual_chart(features=features, start=start, end=end)


    baseline_by_horizon: dict[int, pd.DataFrame] = {}
    h1_with_regimes = baseline_strat.copy()
    if available_regime_columns:
        h1_regimes = aligned[available_regime_columns].reset_index().rename(columns={"index": "date"})
        h1_with_regimes = h1_with_regimes.merge(h1_regimes, on="date", how="left")
    baseline_by_horizon[1] = h1_with_regimes

    horizon_rows: list[dict[str, float | int | str]] = []
    for horizon in sorted(int(h) for h in cfg["horizons"]):
        if horizon == 1:
            strat_h = baseline_by_horizon[1]
        else:
            pred_h = result.predictions[result.predictions["horizon"] == horizon].copy()
            if pred_h.empty:
                print(f"[backtest] strategy_h{horizon}_baseline skipped (no predictions)")
                continue

            pred_h["date"] = pd.to_datetime(pred_h["date"])
            pred_h = pred_h.set_index("date").sort_index()

            aligned_h = pred_h.join(raw[["return_1m", "TB3MS"]], how="inner")
            if available_regime_columns:
                aligned_h = aligned_h.join(features[available_regime_columns], how="left")
            aligned_h["cash_ret"] = (aligned_h["TB3MS"].shift(1) / 100.0) / 12.0
            aligned_h = aligned_h.dropna(subset=["return_1m", "cash_ret"])
            if aligned_h.empty:
                print(f"[backtest] strategy_h{horizon}_baseline skipped (no return/cash overlap)")
                continue

            strat_h = _build_long_only_strategy(aligned=aligned_h, threshold=baseline_threshold)
            if available_regime_columns:
                regime_h = aligned_h[available_regime_columns].reset_index().rename(columns={"index": "date"})
                strat_h = strat_h.merge(regime_h, on="date", how="left")
            baseline_by_horizon[horizon] = strat_h

            strat_h_out = ensure_parent(f"outputs/strategy_h{horizon}_baseline.csv")
            strat_h.to_csv(strat_h_out, index=False)
            print(f"[backtest] wrote strategy -> {strat_h_out}")

        horizon_metric_row = {"horizon": int(horizon)}
        horizon_metric_row.update(_build_strategy_metric_summary(strat_h))
        horizon_rows.append(horizon_metric_row)

    horizon_summary_columns = [
        "horizon",
        "months",
        "final_equity",
        "CAGR",
        "vol",
        "Sharpe",
        "max_drawdown",
        "avg_turnover",
        "hit_rate",
        "invested_fraction",
    ]
    horizon_summary_df = pd.DataFrame(horizon_rows, columns=horizon_summary_columns).sort_values(
        "horizon"
    ).reset_index(drop=True)
    horizon_summary_out = ensure_parent("outputs/strategy_baseline_horizon_summary.csv")
    horizon_summary_df.to_csv(horizon_summary_out, index=False)
    print("[backtest] baseline_strategy_by_horizon")
    print(horizon_summary_df.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))
    print(f"[backtest] wrote baseline horizon summary -> {horizon_summary_out}")

    if missing_regime_columns:
        print(
            "[backtest] regime diagnostics skipped (missing columns: "
            f"{missing_regime_columns})"
        )
    else:
        regime_rows: list[dict[str, float | int | str]] = []
        for horizon, strat_h in sorted(baseline_by_horizon.items()):
            term_spread = pd.to_numeric(strat_h["term_spread_lag1"], errors="coerce")
            fedfunds = pd.to_numeric(strat_h["FEDFUNDS_lag1"], errors="coerce")
            inflation = pd.to_numeric(strat_h["inflation_yoy_lag1"], errors="coerce")

            fed_median = float(fedfunds.dropna().median()) if fedfunds.notna().any() else float("nan")
            inflation_median = float(inflation.dropna().median()) if inflation.notna().any() else float("nan")

            split_specs = [
                {
                    "split": "term_spread_lag1_sign",
                    "split_median": float("nan"),
                    "groups": [
                        ("term_spread_lag1 > 0", term_spread > 0.0),
                        ("term_spread_lag1 <= 0", term_spread <= 0.0),
                    ],
                },
                {
                    "split": "FEDFUNDS_lag1_median",
                    "split_median": fed_median,
                    "groups": [
                        ("FEDFUNDS_lag1 > median", fedfunds > fed_median),
                        ("FEDFUNDS_lag1 <= median", fedfunds <= fed_median),
                    ],
                },
                {
                    "split": "inflation_yoy_lag1_median",
                    "split_median": inflation_median,
                    "groups": [
                        ("inflation_yoy_lag1 > median", inflation > inflation_median),
                        ("inflation_yoy_lag1 <= median", inflation <= inflation_median),
                    ],
                },
            ]

            for split_spec in split_specs:
                for regime_name, regime_mask in split_spec["groups"]:
                    subset = strat_h.loc[regime_mask.fillna(False)].copy()
                    row = {
                        "horizon": int(horizon),
                        "split": str(split_spec["split"]),
                        "regime": str(regime_name),
                        "split_median": float(split_spec["split_median"]),
                    }
                    row.update(_build_strategy_metric_summary(subset))
                    regime_rows.append(row)

        regime_summary_columns = [
            "horizon",
            "split",
            "regime",
            "split_median",
            "months",
            "final_equity",
            "CAGR",
            "vol",
            "Sharpe",
            "max_drawdown",
            "avg_turnover",
            "hit_rate",
            "invested_fraction",
        ]
        regime_summary_df = pd.DataFrame(regime_rows, columns=regime_summary_columns).sort_values(
            ["horizon", "split", "regime"]
        ).reset_index(drop=True)
        regime_summary_out = ensure_parent("outputs/strategy_baseline_regime_summary.csv")
        regime_summary_df.to_csv(regime_summary_out, index=False)
        print("[backtest] baseline_regime_splits")
        print(regime_summary_df.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))
        print(f"[backtest] wrote baseline regime summary -> {regime_summary_out}")

    h3_pred = result.predictions[result.predictions["horizon"] == 3].copy()
    if h3_pred.empty:
        print("[backtest] strategy_h3_regime_gate skipped (no horizon=3 predictions)")
    else:
        h3_pred["date"] = pd.to_datetime(h3_pred["date"])
        h3_pred = h3_pred.set_index("date").sort_index()

        aligned_h3 = h3_pred.join(raw[["return_1m", "TB3MS"]], how="inner")
        if set(h3_gate_columns).issubset(set(features.columns)):
            aligned_h3 = aligned_h3.join(features[h3_gate_columns], how="left")
        aligned_h3["cash_ret"] = (aligned_h3["TB3MS"].shift(1) / 100.0) / 12.0
        aligned_h3 = aligned_h3.dropna(subset=["return_1m", "cash_ret"])

        if aligned_h3.empty:
            print("[backtest] strategy_h3_regime_gate skipped (no return/cash overlap)")
        else:
            missing_gate_cols = [col for col in h3_gate_columns if col not in aligned_h3.columns]
            if missing_gate_cols:
                raise RuntimeError(
                    "Missing required regime columns for --regime-gate in h3 strategy: "
                    f"{missing_gate_cols}"
                )

            gate_mask, fed_median_h3, inflation_median_h3 = _build_h3_regime_gate_mask(
                aligned_h3=aligned_h3,
                regime_gate=regime_gate,
            )

            gated_h3 = _build_long_only_strategy(
                aligned=aligned_h3,
                threshold=baseline_threshold,
                gate_mask=gate_mask,
            )

            gate_series = pd.Series(True, index=aligned_h3.index, dtype=bool) if gate_mask is None else gate_mask
            gate_flags = pd.DataFrame(
                {
                    "date": aligned_h3.index,
                    "gate_condition": gate_series.astype(float).to_numpy(),
                    "FEDFUNDS_lag1": pd.to_numeric(aligned_h3["FEDFUNDS_lag1"], errors="coerce").to_numpy(),
                    "inflation_yoy_lag1": pd.to_numeric(aligned_h3["inflation_yoy_lag1"], errors="coerce").to_numpy(),
                }
            )
            gated_h3 = gated_h3.merge(gate_flags, on="date", how="left")
            gated_h3["regime_gate"] = regime_gate

            gated_h3_out = ensure_parent(f"outputs/strategy_h3_regime_gate_{regime_gate}.csv")
            gated_h3.to_csv(gated_h3_out, index=False)

            gate_summary = {
                "model": model_name,
                "horizon": 3,
                "regime_gate": regime_gate,
                "fedfunds_median_h3": fed_median_h3,
                "inflation_median_h3": inflation_median_h3,
            }
            gate_summary.update(_build_strategy_metric_summary(gated_h3))

            gate_summary_columns = [
                "model",
                "horizon",
                "regime_gate",
                "fedfunds_median_h3",
                "inflation_median_h3",
                "months",
                "final_equity",
                "CAGR",
                "vol",
                "Sharpe",
                "max_drawdown",
                "avg_turnover",
                "hit_rate",
                "invested_fraction",
            ]
            gate_summary_df = pd.DataFrame([gate_summary], columns=gate_summary_columns)
            gate_summary_out = ensure_parent(f"outputs/strategy_h3_regime_gate_summary_{regime_gate}.csv")
            gate_summary_df.to_csv(gate_summary_out, index=False)

            print("[backtest] strategy_h3_regime_gate")
            print(
                gate_summary_df.to_string(index=False, float_format=lambda x: f"{x:0.6f}")
            )
            print(f"[backtest] wrote h3 regime-gated strategy -> {gated_h3_out}")
            print(f"[backtest] wrote h3 regime-gate summary -> {gate_summary_out}")

            h1_pred = result.predictions[result.predictions["horizon"] == 1].copy()
            if h1_pred.empty:
                raise RuntimeError("No horizon=1 predictions found; cannot compute h3 confirmation strategy.")
            h1_pred["date"] = pd.to_datetime(h1_pred["date"])
            h1_pred = h1_pred.set_index("date").sort_index()

            signal_h3 = (pd.to_numeric(aligned_h3["y_pred"], errors="coerce") > 0.0).astype(float)
            signal_h1 = (pd.to_numeric(h1_pred["y_pred"], errors="coerce") > 0.0).astype(float)
            signal_h1 = signal_h1.reindex(aligned_h3.index).fillna(0.0)
            combined_signal = ((signal_h3 > 0.0) & (signal_h1 > 0.0)).astype(float)

            h3_baseline_for_compare = _build_long_only_strategy(
                aligned=aligned_h3,
                threshold=baseline_threshold,
            )
            h3_baseline_for_compare["signal_h3"] = signal_h3.to_numpy(dtype=float)
            h3_baseline_for_compare["signal_h1"] = signal_h1.to_numpy(dtype=float)
            h3_baseline_for_compare["combined_signal"] = combined_signal.to_numpy(dtype=float)
            h3_baseline_for_compare["strategy_type"] = "baseline_h3"

            summary_rows_h3 = []
            baseline_summary_h3 = {
                "model": model_name,
                "horizon": 3,
                "strategy_type": "baseline_h3",
                "h3_confirm_with_h1": bool(h3_confirm_with_h1),
                "h3_smooth_scale": h3_smooth_scale,
                "h3_floor_weight": h3_floor_weight,
            }
            baseline_summary_h3.update(_build_strategy_metric_summary(h3_baseline_for_compare))
            summary_rows_h3.append(baseline_summary_h3)

            if h3_confirm_with_h1:
                h3_confirm = _build_strategy_from_signal(
                    aligned=aligned_h3,
                    signal=combined_signal,
                )
                h3_confirm["signal_h3"] = signal_h3.to_numpy(dtype=float)
                h3_confirm["signal_h1"] = signal_h1.to_numpy(dtype=float)
                h3_confirm["combined_signal"] = combined_signal.to_numpy(dtype=float)
                h3_confirm["strategy_type"] = "h3_confirm_h1"

                h3_confirm_out = ensure_parent("outputs/strategy_h3_confirm_h1.csv")
                h3_confirm.to_csv(h3_confirm_out, index=False)
                print(f"[backtest] wrote h3 confirm strategy -> {h3_confirm_out}")

                confirm_summary_h3 = {
                    "model": model_name,
                    "horizon": 3,
                    "strategy_type": "h3_confirm_h1",
                    "h3_confirm_with_h1": bool(h3_confirm_with_h1),
                    "h3_smooth_scale": h3_smooth_scale,
                    "h3_floor_weight": h3_floor_weight,
                }
                confirm_summary_h3.update(_build_strategy_metric_summary(h3_confirm))
                summary_rows_h3.append(confirm_summary_h3)

            if h3_smooth_scale is not None:
                smooth_signal = _build_h3_smooth_signal(
                    y_pred_h3=aligned_h3["y_pred"],
                    scale=h3_smooth_scale,
                )
                h3_smooth = _build_strategy_from_signal(
                    aligned=aligned_h3,
                    signal=smooth_signal,
                )
                h3_smooth["signal_h3"] = signal_h3.to_numpy(dtype=float)
                h3_smooth["signal_h1"] = signal_h1.to_numpy(dtype=float)
                h3_smooth["combined_signal"] = combined_signal.to_numpy(dtype=float)
                h3_smooth["smooth_signal_raw"] = (
                    pd.to_numeric(aligned_h3["y_pred"], errors="coerce") / float(h3_smooth_scale)
                ).to_numpy(dtype=float)
                h3_smooth["smooth_scale"] = float(h3_smooth_scale)
                h3_smooth["strategy_type"] = "h3_smooth"

                scale_tag = str(f"{float(h3_smooth_scale):0.4f}").replace(".", "p")
                h3_smooth_out = ensure_parent(f"outputs/strategy_h3_smooth_scale_{scale_tag}.csv")
                h3_smooth.to_csv(h3_smooth_out, index=False)
                print(f"[backtest] wrote h3 smooth strategy -> {h3_smooth_out}")

                smooth_summary_h3 = {
                    "model": model_name,
                    "horizon": 3,
                    "strategy_type": "h3_smooth",
                    "h3_confirm_with_h1": bool(h3_confirm_with_h1),
                    "h3_smooth_scale": float(h3_smooth_scale),
                    "h3_floor_weight": h3_floor_weight,
                }
                smooth_summary_h3.update(_build_strategy_metric_summary(h3_smooth))
                summary_rows_h3.append(smooth_summary_h3)

            if h3_floor_weight is not None:
                h3_floor = _apply_weight_floor_overlay(
                    strat=h3_baseline_for_compare,
                    floor_weight=float(h3_floor_weight),
                )
                h3_floor["floor_weight"] = float(h3_floor_weight)
                h3_floor["strategy_type"] = "h3_floor"

                floor_tag = str(f"{float(h3_floor_weight):0.4f}").replace(".", "p")
                h3_floor_out = ensure_parent(f"outputs/strategy_h3_floor_weight_{floor_tag}.csv")
                h3_floor.to_csv(h3_floor_out, index=False)
                print(f"[backtest] wrote h3 floor strategy -> {h3_floor_out}")

                floor_summary_h3 = {
                    "model": model_name,
                    "horizon": 3,
                    "strategy_type": "h3_floor",
                    "h3_confirm_with_h1": bool(h3_confirm_with_h1),
                    "h3_smooth_scale": h3_smooth_scale,
                    "h3_floor_weight": float(h3_floor_weight),
                }
                floor_summary_h3.update(_build_strategy_metric_summary(h3_floor))
                summary_rows_h3.append(floor_summary_h3)

            h3_compare_cols = [
                "model",
                "horizon",
                "strategy_type",
                "h3_confirm_with_h1",
                "h3_smooth_scale",
                "h3_floor_weight",
                "months",
                "final_equity",
                "CAGR",
                "vol",
                "Sharpe",
                "max_drawdown",
                "avg_turnover",
                "total_turnover",
                "hit_rate",
                "invested_fraction",
            ]
            h3_compare_df = pd.DataFrame(summary_rows_h3, columns=h3_compare_cols)
            h3_compare_out = ensure_parent("outputs/strategy_h3_confirmation_summary.csv")
            h3_compare_df.to_csv(h3_compare_out, index=False)
            print("[backtest] h3_confirmation_summary")
            print(h3_compare_df.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))
            print(f"[backtest] wrote h3 confirmation summary -> {h3_compare_out}")


    if threshold_grid:
        thresholds = _parse_threshold_grid(threshold_grid)
        sweep_columns = [
            "model",
            "horizon",
            "start_date",
            "end_date",
            "threshold",
            "final_equity_strat",
            "final_equity_bh",
            "CAGR_strat",
            "CAGR_bh",
            "vol_strat",
            "vol_bh",
            "Sharpe_strat",
            "Sharpe_bh",
            "max_drawdown_strat",
            "max_drawdown_bh",
            "hit_rate_strat",
            "hit_rate_bh",
            "avg_turnover",
            "total_turnover",
            "invested_fraction",
        ]

        sweep_rows: list[dict[str, float | int | str]] = []
        for threshold in thresholds:
            sweep_strat = _build_long_only_strategy(aligned=aligned, threshold=threshold)
            summary = _build_run_summary(
                strat=sweep_strat,
                model_name=model_name,
                horizon=1,
                strategy_name="sweep",
                strategy_type="binary",
                threshold=threshold,
            )
            sweep_rows.append({col: summary[col] for col in sweep_columns})

        sweep_out = ensure_parent("outputs/strategy_threshold_sweep.csv")
        pd.DataFrame(sweep_rows, columns=sweep_columns).to_csv(sweep_out, index=False)
        print(f"[backtest] wrote threshold sweep -> {sweep_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leakage-safe monthly walk-forward scaffold")
    parser.add_argument("command", choices=["data", "features", "targets", "backtest"], help="Pipeline step")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default="ridge",
        help="Model type: ridge or hgb",
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=None,
        help="Optional ridge pruning mode: rank by |coef| and keep top-k features.",
    )
    parser.add_argument(
        "--use-residual-method",
        action="store_true",
        help="Enable leakage-safe residual_method feature in the feature set.",
    )
    parser.add_argument(
        "--residual-warmup",
        type=int,
        default=60,
        help="Minimum monthly observations for expanding residual_method trend fit.",
    )
    parser.add_argument(
        "--use-feature-batch-1",
        action="store_true",
        help="Enable optional candidate feature batch 1 in the feature set.",
    )
    parser.add_argument(
        "--use-feature-batch-2",
        action="store_true",
        help="Enable optional macro/rates feature batch 2 in the feature set.",
    )
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=0.0,
        help="Threshold for alternative long-only rule (weight=1 if y_pred > threshold else 0).",
    )
    parser.add_argument(
        "--tier-threshold",
        type=float,
        default=0.0025,
        help="Tier cutoff for tiered sizing (0->0.5->1.0 mapping on y_pred).",
    )
    parser.add_argument(
        "--threshold-grid",
        type=str,
        default=None,
        help="Optional comma-separated thresholds to sweep on the same prediction stream.",
    )
    parser.add_argument(
        "--regime-gate",
        type=str,
        default="none",
        choices=list(REGIME_GATE_CHOICES),
        help=(
            "Optional h3-only trade gate applied after 1M signal lag: "
            "none|fedfunds_high|inflation_low|both."
        ),
    )
    parser.add_argument(
        "--h3-confirm-with-h1",
        action="store_true",
        help="For h3 strategy only, require both h3 and h1 predictions > 0 before 1M lagged execution.",
    )
    parser.add_argument(
        "--h3-smooth-scale",
        type=float,
        default=None,
        help="Optional h3-only smooth long/cash sizing scale: weight=clip(y_pred_h3/scale,0,1), then 1M lag.",
    )
    parser.add_argument(
        "--h3-floor-weight",
        type=float,
        default=None,
        help="Optional h3-only floor overlay: final weight=max(floor_weight, baseline_lagged_binary_weight).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.command == "data":
        run_data(cfg)
    elif args.command == "features":
        run_features(
            cfg,
            use_residual_method=args.use_residual_method,
            residual_warmup=args.residual_warmup,
            use_feature_batch_1=args.use_feature_batch_1,
            use_feature_batch_2=args.use_feature_batch_2,
        )
    elif args.command == "targets":
        run_targets(cfg)
    elif args.command == "backtest":
        run_backtest(
            cfg,
            model_name=args.model,
            start=args.start,
            end=args.end,
            signal_threshold=args.signal_threshold,
            tier_threshold=args.tier_threshold,
            threshold_grid=args.threshold_grid,
            use_residual_method=args.use_residual_method,
            residual_warmup=args.residual_warmup,
            use_feature_batch_1=args.use_feature_batch_1,
            use_feature_batch_2=args.use_feature_batch_2,
            top_k_features=args.top_k_features,
            regime_gate=args.regime_gate,
            h3_confirm_with_h1=args.h3_confirm_with_h1,
            h3_smooth_scale=args.h3_smooth_scale,
            h3_floor_weight=args.h3_floor_weight,
        )


if __name__ == "__main__":
    main()
