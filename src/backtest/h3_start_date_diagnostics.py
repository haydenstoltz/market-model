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
class StageSummary:
    stage: str
    earliest_date_present: str
    earliest_all_required_non_null: str
    row_count: int
    required_column_count: int
    blocking_columns: str
    delay_months_to_all_required: float
    constraint_classification: str
    notes: str


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_iso_date(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return ""
    return pd.Timestamp(ts).date().isoformat()


def _month_diff(start: pd.Timestamp | None, end: pd.Timestamp | None) -> float:
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return float("nan")
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    return float((e.year - s.year) * 12 + (e.month - s.month))


def _stage_summary(
    stage: str,
    df: pd.DataFrame,
    required_cols: list[str],
    classification: str,
    notes: str,
) -> StageSummary:
    if df.empty:
        return StageSummary(
            stage=stage,
            earliest_date_present="",
            earliest_all_required_non_null="",
            row_count=0,
            required_column_count=len(required_cols),
            blocking_columns="",
            delay_months_to_all_required=float("nan"),
            constraint_classification=classification,
            notes=notes,
        )

    idx = pd.to_datetime(df.index)
    earliest_present = pd.Timestamp(idx.min())

    usable_date: pd.Timestamp | None = None
    blocking_cols: list[str] = []
    if required_cols:
        sub = df[required_cols].copy()
        mask = sub.notna().all(axis=1)
        if bool(mask.any()):
            usable_date = pd.Timestamp(mask[mask].index[0])
        else:
            usable_date = None
            blocking_cols = sorted(required_cols)

        # Bottlenecks should only include columns that directly set the usable start.
        if usable_date is not None:
            first_valid_map = {c: sub[c].first_valid_index() for c in required_cols}
            first_valid_dates = [pd.Timestamp(v) for v in first_valid_map.values() if v is not None]
            if first_valid_dates:
                max_first_valid = max(first_valid_dates)
                delay_months = _month_diff(earliest_present, usable_date)
                if np.isfinite(delay_months) and delay_months > 0.0:
                    blocking_cols = sorted(
                        [c for c, v in first_valid_map.items() if v is None or pd.Timestamp(v) == max_first_valid]
                    )
                else:
                    blocking_cols = []

    return StageSummary(
        stage=stage,
        earliest_date_present=_safe_iso_date(earliest_present),
        earliest_all_required_non_null=_safe_iso_date(usable_date),
        row_count=int(len(df)),
        required_column_count=int(len(required_cols)),
        blocking_columns=";".join(blocking_cols),
        delay_months_to_all_required=_month_diff(earliest_present, usable_date),
        constraint_classification=classification,
        notes=notes,
    )


def _collect_blocking_details(
    stage: str,
    df: pd.DataFrame,
    required_cols: list[str],
) -> pd.DataFrame:
    cols = [
        "stage",
        "column",
        "first_non_null_date",
        "non_null_count",
        "delay_months_from_stage_start",
        "blocks_stage_start",
    ]
    if df.empty or not required_cols:
        return pd.DataFrame(columns=cols)

    idx = pd.to_datetime(df.index)
    stage_start = pd.Timestamp(idx.min())
    sub = df[required_cols].copy()
    mask = sub.notna().all(axis=1)
    stage_usable = pd.Timestamp(mask[mask].index[0]) if bool(mask.any()) else None
    first_valid_map = {c: sub[c].first_valid_index() for c in required_cols}
    first_valid_dates = [pd.Timestamp(v) for v in first_valid_map.values() if v is not None]
    max_first_valid = max(first_valid_dates) if first_valid_dates else None

    rows: list[dict[str, float | int | str | bool]] = []
    for col in required_cols:
        series = pd.to_numeric(sub[col], errors="coerce") if pd.api.types.is_numeric_dtype(sub[col]) else sub[col]
        first_valid = first_valid_map[col]
        first_valid_ts = pd.Timestamp(first_valid) if first_valid is not None else None
        delay = _month_diff(stage_start, first_valid_ts)
        stage_delay = _month_diff(stage_start, stage_usable)
        blocks = bool(
            first_valid_ts is None
            or (
                max_first_valid is not None
                and stage_usable is not None
                and np.isfinite(stage_delay)
                and stage_delay > 0.0
                and first_valid_ts == max_first_valid
            )
        )
        rows.append(
            {
                "stage": stage,
                "column": col,
                "first_non_null_date": _safe_iso_date(first_valid_ts),
                "non_null_count": int(series.notna().sum()),
                "delay_months_from_stage_start": delay,
                "blocks_stage_start": blocks,
            }
        )

    out = pd.DataFrame(rows, columns=cols).sort_values(
        ["stage", "delay_months_from_stage_start", "column"],
        ascending=[True, False, True],
    )
    return out.reset_index(drop=True)


def _run_backtest(project_root: Path) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "src.cli",
        "backtest",
        "--model",
        "ridge",
        "--start",
        "2000-01-01",
    ]
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, check=False)
    log_out = project_root / "outputs" / "h3_start_date_diagnostics_backtest.log"
    log_out.write_text((result.stdout or "") + ("\n" + result.stderr if result.stderr else ""), encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Backtest failed (returncode={result.returncode}). See {log_out}.")
    return log_out


def build_h3_start_date_diagnostics(
    project_root: Path,
    config_path: Path,
    start_date: str,
    end_date: str | None,
    run_backtest: bool = True,
) -> dict[str, Path]:
    if run_backtest:
        _run_backtest(project_root=project_root)

    cfg = _load_config(config_path=config_path)
    paths = cfg["paths"]
    train_min_periods = int(cfg.get("train_min_periods", 36))
    horizon = 3

    raw = pd.read_csv(project_root / paths["raw_data"], parse_dates=["date"], index_col="date").sort_index()
    features = pd.read_csv(project_root / paths["features"], parse_dates=["date"], index_col="date").sort_index()
    targets = pd.read_csv(project_root / paths["targets"], parse_dates=["date"], index_col="date").sort_index()
    predictions = pd.read_csv(project_root / paths["predictions"], parse_dates=["date"])

    h3_strategy_path = project_root / "outputs" / "strategy_h3_baseline.csv"
    if not h3_strategy_path.exists():
        raise RuntimeError(f"Expected strategy output missing: {h3_strategy_path}")
    strategy_h3 = pd.read_csv(h3_strategy_path, parse_dates=["date"]).set_index("date").sort_index()

    market_cols = [c for c in ["close", "return_1m"] if c in raw.columns]
    market_df = raw[market_cols].copy()
    merged_df = raw.copy()

    target_h3_col = "target_h3"
    if target_h3_col not in targets.columns:
        raise RuntimeError("targets.csv missing required target_h3 column.")

    joined_full = (
        features.reset_index()
        .merge(targets.reset_index(), on="date", how="inner")
        .sort_values("date")
        .set_index("date")
    )
    feature_cols = [c for c in joined_full.columns if not c.startswith("target_h")]
    feature_cols = [c for c in feature_cols if joined_full[c].notna().any()]
    required_model_cols = list(feature_cols) + [target_h3_col]

    joined_run = joined_full.copy()
    if start_date:
        joined_run = joined_run[joined_run.index >= pd.to_datetime(start_date)]
    if end_date:
        joined_run = joined_run[joined_run.index <= pd.to_datetime(end_date)]

    h3_predictions = predictions[predictions["horizon"] == horizon].copy()
    h3_predictions = h3_predictions.sort_values("date").set_index("date")

    stage_rows: list[StageSummary] = []
    stage_rows.append(
        _stage_summary(
            stage="raw_monthly_market",
            df=market_df,
            required_cols=list(market_df.columns),
            classification="unavoidable_warmup",
            notes="Market close + return_1m (return_1m requires one prior month).",
        )
    )
    stage_rows.append(
        _stage_summary(
            stage="merged_macro_data",
            df=merged_df,
            required_cols=list(merged_df.columns),
            classification="data_availability_constraint",
            notes="Merged market+macro monthly frame written to raw_data path.",
        )
    )
    stage_rows.append(
        _stage_summary(
            stage="engineered_features",
            df=features,
            required_cols=list(features.columns),
            classification="data_availability_constraint",
            notes="Baseline feature matrix used by backtest.",
        )
    )
    stage_rows.append(
        _stage_summary(
            stage="engineered_targets",
            df=targets,
            required_cols=list(targets.columns),
            classification="unavoidable_warmup",
            notes="Residual-change targets with ma_window warmup and horizon shift.",
        )
    )
    stage_rows.append(
        _stage_summary(
            stage="modeling_frame_h3_full",
            df=joined_full,
            required_cols=required_model_cols,
            classification="data_availability_constraint",
            notes="Features+targets inner-joined before user start/end filters.",
        )
    )
    stage_rows.append(
        _stage_summary(
            stage="modeling_frame_h3_run_window",
            df=joined_run,
            required_cols=required_model_cols,
            classification="overly_conservative_filter",
            notes=f"Walkforward frame after --start={start_date} filter.",
        )
    )
    stage_rows.append(
        _stage_summary(
            stage="predictions_h3",
            df=h3_predictions,
            required_cols=["y_true", "y_pred", "error"],
            classification="unavoidable_warmup",
            notes="Walkforward predictions for horizon=3 after train_min_periods.",
        )
    )
    stage_rows.append(
        _stage_summary(
            stage="strategy_h3_output",
            df=strategy_h3,
            required_cols=[
                "signal",
                "weight",
                "cash_ret",
                "turnover",
                "cost",
                "strat_ret",
                "bh_ret",
            ],
            classification="unavoidable_warmup",
            notes="h3 baseline strategy series after lagged execution + cash return alignment.",
        )
    )

    first_run_date = pd.Timestamp(joined_run.index.min()) if not joined_run.empty else None
    first_h3_pred_date = pd.Timestamp(h3_predictions.index.min()) if not h3_predictions.empty else None
    first_h3_strategy_date = pd.Timestamp(strategy_h3.index.min()) if not strategy_h3.empty else None
    train_min_candidate_date = (
        pd.Timestamp(joined_run.index[train_min_periods])
        if len(joined_run) > train_min_periods
        else None
    )

    rule_rows = [
        {
            "stage": "rule_start_filter",
            "earliest_date_present": _safe_iso_date(pd.Timestamp(joined_full.index.min()) if not joined_full.empty else None),
            "earliest_all_required_non_null": _safe_iso_date(first_run_date),
            "row_count": int(len(joined_run)),
            "required_column_count": 0,
            "blocking_columns": "--start",
            "delay_months_to_all_required": _month_diff(
                pd.Timestamp(joined_full.index.min()) if not joined_full.empty else None,
                first_run_date,
            ),
            "constraint_classification": "overly_conservative_filter",
            "notes": f"Explicit CLI filter --start={start_date}.",
        },
        {
            "stage": "rule_train_min_periods",
            "earliest_date_present": _safe_iso_date(first_run_date),
            "earliest_all_required_non_null": _safe_iso_date(first_h3_pred_date),
            "row_count": int(train_min_periods),
            "required_column_count": 0,
            "blocking_columns": "train_min_periods",
            "delay_months_to_all_required": _month_diff(first_run_date, first_h3_pred_date),
            "constraint_classification": "unavoidable_warmup",
            "notes": (
                f"Expanding walkforward requires {train_min_periods} in-sample rows "
                "before first out-of-sample prediction."
            ),
        },
        {
            "stage": "rule_cash_return_shift_dropna",
            "earliest_date_present": _safe_iso_date(first_h3_pred_date),
            "earliest_all_required_non_null": _safe_iso_date(first_h3_strategy_date),
            "row_count": 1,
            "required_column_count": 0,
            "blocking_columns": "cash_ret",
            "delay_months_to_all_required": _month_diff(first_h3_pred_date, first_h3_strategy_date),
            "constraint_classification": "unavoidable_warmup",
            "notes": "cash_ret uses TB3MS.shift(1) on the prediction-aligned frame; first row is dropped.",
        },
        {
            "stage": "rule_train_min_candidate_check",
            "earliest_date_present": _safe_iso_date(train_min_candidate_date),
            "earliest_all_required_non_null": _safe_iso_date(first_h3_pred_date),
            "row_count": 1,
            "required_column_count": 0,
            "blocking_columns": "",
            "delay_months_to_all_required": _month_diff(train_min_candidate_date, first_h3_pred_date),
            "constraint_classification": "sanity_check",
            "notes": "Checks whether first possible train_min date matches first produced h3 prediction.",
        },
    ]

    diag_df = pd.DataFrame([s.__dict__ for s in stage_rows] + rule_rows)

    blocking_frames = [
        _collect_blocking_details(stage="raw_monthly_market", df=market_df, required_cols=list(market_df.columns)),
        _collect_blocking_details(stage="merged_macro_data", df=merged_df, required_cols=list(merged_df.columns)),
        _collect_blocking_details(stage="engineered_features", df=features, required_cols=list(features.columns)),
        _collect_blocking_details(stage="engineered_targets", df=targets, required_cols=list(targets.columns)),
        _collect_blocking_details(stage="modeling_frame_h3_full", df=joined_full, required_cols=required_model_cols),
        _collect_blocking_details(stage="modeling_frame_h3_run_window", df=joined_run, required_cols=required_model_cols),
    ]
    blocking_df = pd.concat(blocking_frames, ignore_index=True)

    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    diag_out = outputs_dir / "h3_start_date_diagnostics.csv"
    blocking_out = outputs_dir / "h3_blocking_columns.csv"
    diag_df.to_csv(diag_out, index=False)
    blocking_df.to_csv(blocking_out, index=False)

    return {"diagnostics": diag_out, "blocking_columns": blocking_out}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose why h3 strategy history starts around 2010.")
    parser.add_argument("--project-root", type=str, default=".", help="Project root path.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config path relative to project root.")
    parser.add_argument("--start", type=str, default="2000-01-01", help="Backtest start date used for diagnostics.")
    parser.add_argument("--end", type=str, default=None, help="Optional backtest end date.")
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Use existing outputs from previous backtest run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    artifacts = build_h3_start_date_diagnostics(
        project_root=project_root,
        config_path=(project_root / args.config),
        start_date=str(args.start),
        end_date=(None if args.end in (None, "") else str(args.end)),
        run_backtest=not bool(args.skip_backtest),
    )
    print("[h3-start-date-diagnostics] wrote:")
    for key, path in artifacts.items():
        print(f" - {key}: {path}")


if __name__ == "__main__":
    main()
