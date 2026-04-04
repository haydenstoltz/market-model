from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


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

MACRO_FEATURES: tuple[str, ...] = (
    "FEDFUNDS_lag1",
    "FEDFUNDS_lag3",
    "TB3MS_lag1",
    "TB3MS_lag3",
    "GS10_lag1",
    "GS10_lag3",
    "term_spread_lag1",
    "yc_10y_2y_lag1",
    "CPIAUCSL_lag3",
    "inflation_yoy_lag1",
    "INDPRO_lag1",
    "INDPRO_lag3",
    "credit_spread_lag1",
    "AAA_lag1",
    "AAA_lag3",
)


def _load_config(project_root: Path, config_path: str) -> dict:
    cfg_path = (project_root / config_path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_backtest(project_root: Path, start_date: str) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "src.cli",
        "backtest",
        "--model",
        "ridge",
        "--start",
        str(start_date),
    ]
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path = project_root / "outputs" / "h3_macro_state_backtest.log"
    log_path.write_text(
        (result.stdout or "") + ("\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Backtest failed (rc={result.returncode}). See {log_path}")
    return log_path


def _assign_subperiod(dates: pd.Series) -> pd.Series:
    out = pd.Series("outside_range", index=dates.index, dtype="object")
    dt = pd.to_datetime(dates, errors="coerce")
    for spec in SUBPERIODS:
        start_ts = pd.Timestamp(spec.start)
        end_ts = pd.Timestamp.max if spec.end is None else pd.Timestamp(spec.end)
        mask = (dt >= start_ts) & (dt <= end_ts)
        out.loc[mask] = spec.label
    return out


def _compute_confusion_bucket(df: pd.DataFrame) -> pd.Series:
    market_up = pd.to_numeric(df["market_return"], errors="coerce") > 0.0
    invested = pd.to_numeric(df["final_weight"], errors="coerce") > 0.0

    labels = np.where(
        market_up & invested,
        "TP",
        np.where(
            market_up & (~invested),
            "FN",
            np.where((~market_up) & invested, "FP", "TN"),
        ),
    )
    return pd.Series(labels, index=df.index, dtype="object")


def _summarize_feature_stats(
    df: pd.DataFrame,
    group_cols: list[str],
    feature_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(group_cols, dropna=False)
    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_map = dict(zip(group_cols, group_key))
        for feature in feature_cols:
            series = pd.to_numeric(group_df[feature], errors="coerce").dropna().astype(float)
            rows.append(
                {
                    **group_map,
                    "feature_name": feature,
                    "count": int(series.shape[0]),
                    "mean": float(series.mean()) if not series.empty else float("nan"),
                    "median": float(series.median()) if not series.empty else float("nan"),
                    "std": float(series.std(ddof=0)) if not series.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_h3_macro_state_diagnostics(
    project_root: Path,
    config_path: str = "config.yaml",
    start_date: str = "1976-10-31",
    run_backtest: bool = True,
) -> dict[str, Path]:
    outputs = project_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    if run_backtest:
        _run_backtest(project_root=project_root, start_date=start_date)

    cfg = _load_config(project_root=project_root, config_path=config_path)
    features_path = project_root / cfg["paths"]["features"]

    strat_path = outputs / "strategy_h3_baseline.csv"
    pred_path = outputs / "predictions.csv"

    if not strat_path.exists():
        raise RuntimeError(f"Missing strategy file: {strat_path}")
    if not pred_path.exists():
        raise RuntimeError(f"Missing predictions file: {pred_path}")
    if not features_path.exists():
        raise RuntimeError(f"Missing features file: {features_path}")

    strat = pd.read_csv(strat_path)
    preds = pd.read_csv(pred_path)
    feats = pd.read_csv(features_path)

    for frame_name, frame in (("strategy", strat), ("predictions", preds), ("features", feats)):
        if "date" not in frame.columns:
            raise RuntimeError(f"{frame_name} file missing required 'date' column.")

    strat["date"] = pd.to_datetime(strat["date"], errors="coerce")
    preds["date"] = pd.to_datetime(preds["date"], errors="coerce")
    feats["date"] = pd.to_datetime(feats["date"], errors="coerce")

    strat = strat.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    preds = preds.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    feats = feats.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    pred_h3 = preds[preds["horizon"] == 3][["date", "y_pred"]].copy()
    pred_h3 = pred_h3.rename(columns={"y_pred": "y_pred_h3"})

    missing_features = [feature for feature in MACRO_FEATURES if feature not in feats.columns]
    if missing_features:
        raise RuntimeError(f"Missing required macro features in features file: {missing_features}")

    state = strat.merge(pred_h3, on="date", how="left")
    overlapping_macro_cols = [feature for feature in MACRO_FEATURES if feature in state.columns]
    if overlapping_macro_cols:
        state = state.drop(columns=overlapping_macro_cols)
    state = state.merge(feats[["date", *MACRO_FEATURES]], on="date", how="left")

    state["signal_h3"] = pd.to_numeric(state["signal"], errors="coerce")
    state["final_weight"] = pd.to_numeric(state["weight"], errors="coerce")
    state["market_return"] = pd.to_numeric(state["bh_ret"], errors="coerce")
    state["cash_return"] = pd.to_numeric(state["cash_ret"], errors="coerce")
    state["strategy_return"] = pd.to_numeric(state["strat_ret"], errors="coerce")

    state["position_state"] = np.where(state["final_weight"] > 0.0, "invested", "cash")
    state["confusion_bucket"] = _compute_confusion_bucket(state)
    state["subperiod"] = _assign_subperiod(state["date"])

    output_cols = [
        "date",
        "y_pred_h3",
        "signal_h3",
        "final_weight",
        "market_return",
        "cash_return",
        "strategy_return",
        "position_state",
        "confusion_bucket",
        "subperiod",
        *MACRO_FEATURES,
    ]
    state_out = state[output_cols].copy()

    by_position = _summarize_feature_stats(
        df=state_out,
        group_cols=["position_state"],
        feature_cols=list(MACRO_FEATURES),
    ).sort_values(["position_state", "feature_name"]).reset_index(drop=True)

    by_bucket = _summarize_feature_stats(
        df=state_out,
        group_cols=["confusion_bucket"],
        feature_cols=list(MACRO_FEATURES),
    ).sort_values(["confusion_bucket", "feature_name"]).reset_index(drop=True)

    by_subperiod = _summarize_feature_stats(
        df=state_out[state_out["subperiod"] != "outside_range"].copy(),
        group_cols=["subperiod"],
        feature_cols=list(MACRO_FEATURES),
    ).sort_values(["subperiod", "feature_name"]).reset_index(drop=True)

    state_table_out = outputs / "strategy_h3_macro_state_table.csv"
    by_position_out = outputs / "strategy_h3_macro_state_summary_by_position.csv"
    by_bucket_out = outputs / "strategy_h3_macro_state_summary_by_confusion_bucket.csv"
    by_subperiod_out = outputs / "strategy_h3_macro_state_summary_by_subperiod.csv"

    state_out.to_csv(state_table_out, index=False)
    by_position.to_csv(by_position_out, index=False)
    by_bucket.to_csv(by_bucket_out, index=False)
    by_subperiod.to_csv(by_subperiod_out, index=False)

    return {
        "state_table": state_table_out,
        "summary_by_position": by_position_out,
        "summary_by_confusion_bucket": by_bucket_out,
        "summary_by_subperiod": by_subperiod_out,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Macro-state diagnostics for h3 baseline strategy.")
    parser.add_argument("--project-root", type=str, default=".", help="Project root path.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path.")
    parser.add_argument("--start-date", type=str, default="1976-10-31", help="Backtest start date.")
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip rerunning backtest; use existing outputs/strategy_h3_baseline.csv and outputs/predictions.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    artifacts = build_h3_macro_state_diagnostics(
        project_root=project_root,
        config_path=str(args.config),
        start_date=str(args.start_date),
        run_backtest=not bool(args.skip_backtest),
    )
    print("[h3-macro-state] wrote:")
    for name, path in artifacts.items():
        print(f" - {name}: {path}")


if __name__ == "__main__":
    main()
