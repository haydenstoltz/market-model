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


def _compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    y_true_arr = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int).to_numpy()
    y_pred_arr = pd.to_numeric(y_pred, errors="coerce").fillna(0).astype(int).to_numpy()

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _assign_subperiod(date_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(date_series, errors="coerce")
    out = pd.Series("outside_range", index=dt.index, dtype="object")
    for spec in SUBPERIODS:
        start_ts = pd.Timestamp(spec.start)
        end_ts = pd.Timestamp.max if spec.end is None else pd.Timestamp(spec.end)
        mask = (dt >= start_ts) & (dt <= end_ts)
        out.loc[mask] = spec.label
    return out


def _build_summary_rows(
    y_true: pd.Series,
    y_pred: pd.Series,
    subperiod: pd.Series,
    model_name: str,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []

    full_metrics = _compute_metrics(y_true=y_true, y_pred=y_pred)
    rows.append(
        {
            "model": model_name,
            "sample": "full_history",
            "months": int(len(y_true)),
            **full_metrics,
        }
    )

    for spec in SUBPERIODS:
        mask = subperiod == spec.label
        if int(mask.sum()) == 0:
            continue
        period_metrics = _compute_metrics(y_true=y_true.loc[mask], y_pred=y_pred.loc[mask])
        rows.append(
            {
                "model": model_name,
                "sample": spec.label,
                "months": int(mask.sum()),
                **period_metrics,
            }
        )

    return rows


def _extract_tree_rules(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    sklearn_tree = tree.tree_
    rows: list[dict[str, float | int | str]] = []

    def recurse(node: int, conditions: list[str]) -> None:
        left = sklearn_tree.children_left[node]
        right = sklearn_tree.children_right[node]
        is_leaf = left == right

        if is_leaf:
            value = sklearn_tree.value[node][0]
            total = float(value.sum())
            probs = (value / total) if total > 0 else np.zeros_like(value)
            node_samples = int(sklearn_tree.n_node_samples[node])

            invest_prob = float(probs[1]) if probs.shape[0] > 1 else float("nan")
            invest_count_est = int(round(invest_prob * node_samples)) if np.isfinite(invest_prob) else 0
            cash_count_est = int(node_samples - invest_count_est)

            pred_idx = int(np.argmax(probs)) if total > 0 else 0
            predicted_class = int(tree.classes_[pred_idx]) if hasattr(tree, "classes_") else pred_idx
            rows.append(
                {
                    "rule_path": " and ".join(conditions) if conditions else "all months",
                    "predicted_state": "invested" if predicted_class == 1 else "cash",
                    "node_samples": int(node_samples),
                    "invest_prob": float(invest_prob),
                    "invest_count": int(invest_count_est),
                    "cash_count": int(cash_count_est),
                }
            )
            return

        feature_idx = int(sklearn_tree.feature[node])
        threshold = float(sklearn_tree.threshold[node])
        feature = feature_names[feature_idx]

        recurse(left, [*conditions, f"{feature} <= {threshold:.6f}"])
        recurse(right, [*conditions, f"{feature} > {threshold:.6f}"])

    recurse(0, [])
    return pd.DataFrame(rows)


def build_h3_allocation_proxy(
    project_root: Path,
    max_depth: int = 3,
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
    if "date" not in state.columns:
        raise RuntimeError(f"Missing 'date' column in {state_path}.")
    state["date"] = pd.to_datetime(state["date"], errors="coerce")
    state = state.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    missing_cols = [col for col in PROXY_FEATURES if col not in state.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required proxy features in state table: {missing_cols}")

    if "final_weight" not in state.columns:
        raise RuntimeError("Missing 'final_weight' in h3 macro-state table.")

    work = state.copy()
    work["subperiod"] = _assign_subperiod(work["date"])
    work["invested_true"] = (pd.to_numeric(work["final_weight"], errors="coerce") > 0.0).astype(int)

    model_frame = work[["date", "subperiod", "invested_true", *PROXY_FEATURES]].copy()
    model_frame = model_frame.dropna(subset=list(PROXY_FEATURES)).reset_index(drop=True)

    X = model_frame[list(PROXY_FEATURES)].astype(float)
    y = model_frame["invested_true"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logit = LogisticRegression(max_iter=5000, random_state=42)
    logit.fit(X_scaled, y)
    logit_proba = logit.predict_proba(X_scaled)[:, 1]
    logit_pred = (logit_proba >= 0.5).astype(int)

    tree = DecisionTreeClassifier(max_depth=int(max_depth), random_state=42)
    tree.fit(X, y)
    tree_pred = tree.predict(X).astype(int)

    predictions_out = model_frame[["date", "subperiod", "invested_true", *PROXY_FEATURES]].copy()
    predictions_out["logit_proba_invested"] = logit_proba
    predictions_out["logit_pred_invested"] = logit_pred
    predictions_out["tree_pred_invested"] = tree_pred

    logit_rows = _build_summary_rows(
        y_true=predictions_out["invested_true"],
        y_pred=predictions_out["logit_pred_invested"],
        subperiod=predictions_out["subperiod"],
        model_name="logit_compact_macro",
    )
    logit_summary = pd.DataFrame(logit_rows)

    tree_rows = _build_summary_rows(
        y_true=predictions_out["invested_true"],
        y_pred=predictions_out["tree_pred_invested"],
        subperiod=predictions_out["subperiod"],
        model_name="tree_compact_macro",
    )
    tree_summary = pd.DataFrame(tree_rows)
    tree_summary["tree_max_depth"] = int(max_depth)
    tree_summary["tree_realized_depth"] = int(tree.get_depth())
    tree_summary["tree_num_leaves"] = int(tree.get_n_leaves())

    coef_rows: list[dict[str, float | str]] = []
    coef_rows.append(
        {
            "feature_name": "intercept",
            "coefficient_standardized": float(logit.intercept_[0]),
            "odds_ratio_per_1sd": float(np.exp(logit.intercept_[0])),
            "sign": "positive" if float(logit.intercept_[0]) > 0 else "negative",
        }
    )
    for feature, coef in zip(PROXY_FEATURES, logit.coef_[0]):
        coef_rows.append(
            {
                "feature_name": feature,
                "coefficient_standardized": float(coef),
                "odds_ratio_per_1sd": float(np.exp(coef)),
                "sign": "positive" if float(coef) > 0 else "negative",
            }
        )
    coef_df = pd.DataFrame(coef_rows).sort_values(
        "coefficient_standardized",
        key=lambda s: s.abs(),
        ascending=False,
    ).reset_index(drop=True)

    tree_importance_df = pd.DataFrame(
        {
            "feature_name": list(PROXY_FEATURES),
            "importance": tree.feature_importances_.astype(float),
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    rules_df = _extract_tree_rules(tree=tree, feature_names=list(PROXY_FEATURES))

    logit_out = outputs / "h3_allocation_proxy_logit_summary.csv"
    tree_out = outputs / "h3_allocation_proxy_tree_summary.csv"
    preds_out = outputs / "h3_allocation_proxy_predictions.csv"
    logit_coef_out = outputs / "h3_allocation_proxy_logit_coefficients.csv"
    tree_importance_out = outputs / "h3_allocation_proxy_tree_importance.csv"
    tree_rules_out = outputs / "h3_allocation_proxy_tree_rules.csv"

    logit_summary.to_csv(logit_out, index=False)
    tree_summary.to_csv(tree_out, index=False)
    predictions_out.to_csv(preds_out, index=False)
    coef_df.to_csv(logit_coef_out, index=False)
    tree_importance_df.to_csv(tree_importance_out, index=False)
    rules_df.to_csv(tree_rules_out, index=False)

    return {
        "logit_summary": logit_out,
        "tree_summary": tree_out,
        "predictions": preds_out,
        "logit_coefficients": logit_coef_out,
        "tree_importance": tree_importance_out,
        "tree_rules": tree_rules_out,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact macro proxy diagnostics for h3 allocation behavior.")
    parser.add_argument("--project-root", type=str, default=".", help="Project root path.")
    parser.add_argument("--tree-max-depth", type=int, default=3, help="Decision tree max depth (recommended 2 or 3).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    artifacts = build_h3_allocation_proxy(
        project_root=project_root,
        max_depth=int(args.tree_max_depth),
    )
    print("[h3-allocation-proxy] wrote:")
    for key, path in artifacts.items():
        print(f" - {key}: {path}")


if __name__ == "__main__":
    main()
