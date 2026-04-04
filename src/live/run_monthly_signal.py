from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.cli import run_data, run_features, run_targets
from src.models.baselines import make_model


CHAMPION_SPEC: dict[str, Any] = {
    "model": "ridge",
    "feature_set": "baseline",
    "horizon": 3,
    "signal_rule": "signal_t = 1 if y_pred_h3_t > 0 else 0",
    "execution_rule": "weight_t = signal_(t-1)",
    "start_date": "1976-10-31",
    "train_min_periods": 120,
}

MODEL_VERSION = "ridge_baseline_h3_sign_lag1_start1976-10-31_v1"

BASELINE_FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1m_lag1",
    "ret_3m_lag1",
    "ma_120_lag1",
    "price_to_ma_lag1",
    "CPIAUCSL_lag1",
    "CPIAUCSL_lag3",
    "UNRATE_lag1",
    "UNRATE_lag3",
    "INDPRO_lag1",
    "INDPRO_lag3",
    "FEDFUNDS_lag1",
    "FEDFUNDS_lag3",
    "GS10_lag1",
    "GS10_lag3",
    "GS2_lag1",
    "GS2_lag3",
    "TB3MS_lag1",
    "TB3MS_lag3",
    "BAA_lag1",
    "BAA_lag3",
    "AAA_lag1",
    "AAA_lag3",
    "term_spread_lag1",
    "term_spread_lag3",
    "unemployment_rate_lag1",
    "unemployment_rate_lag3",
    "inflation_mom_lag1",
    "inflation_mom_lag3",
    "yc_10y_3m_lag1",
    "yc_10y_3m_lag3",
    "yc_10y_2y_lag1",
    "yc_10y_2y_lag3",
    "credit_spread_lag1",
    "credit_spread_lag3",
    "inflation_yoy_lag1",
    "inflation_yoy_lag3",
    "ip_yoy_lag1",
    "ip_yoy_lag3",
    "unrate_3m_change_lag1",
    "unrate_3m_change_lag3",
)


@dataclass
class SafetyCheck:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "detail": str(self.detail),
        }


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if path.exists():
        df_old = pd.read_csv(path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(path, index=False)


def _format_month(period: pd.Period) -> str:
    return period.strftime("%Y-%m")


def _parse_signal_month(signal_month: str | None) -> pd.Period | None:
    if signal_month is None:
        return None
    try:
        parsed = datetime.strptime(signal_month, "%Y-%m")
    except ValueError as exc:
        raise RuntimeError("Invalid --signal-month format. Use YYYY-MM, e.g. 2026-02.") from exc
    return pd.Period(parsed, freq="M")


def _parse_as_of_date(as_of_date: str | None) -> pd.Timestamp | None:
    if as_of_date is None:
        return None
    try:
        parsed = datetime.strptime(as_of_date, "%Y-%m-%d")
    except ValueError as exc:
        raise RuntimeError("Invalid --as-of-date format. Use YYYY-MM-DD, e.g. 2026-04-02.") from exc
    return pd.Timestamp(parsed.date())


def _resolve_run_date(as_of_date: str | None) -> tuple[pd.Timestamp, bool]:
    parsed = _parse_as_of_date(as_of_date)
    if parsed is None:
        return pd.Timestamp(datetime.now(UTC).date()), False
    return parsed, True


def _month_end_timestamp(period: pd.Period) -> pd.Timestamp:
    return period.to_timestamp(how="end").normalize()


def _first_business_day_next_month(signal_date: pd.Timestamp) -> pd.Timestamp:
    next_month_start = pd.Timestamp(year=signal_date.year, month=signal_date.month, day=1) + pd.offsets.MonthBegin(1)
    return pd.bdate_range(start=next_month_start, periods=1)[0]


def _execution_window_from_signal_date(signal_date: pd.Timestamp, execution_window_business_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    exec_start = _first_business_day_next_month(signal_date)
    exec_end = pd.bdate_range(start=exec_start, periods=int(execution_window_business_days))[-1]
    return exec_start, exec_end


def _window_position(run_date: pd.Timestamp, exec_start: pd.Timestamp, exec_end: pd.Timestamp) -> str:
    if run_date < exec_start:
        return "before"
    if run_date > exec_end:
        return "after"
    return "inside"


def _valid_inference_rows(features: pd.DataFrame, start_date: str) -> tuple[pd.DataFrame, list[str]]:
    feat = features.copy()
    feat.index = pd.to_datetime(feat.index)
    feat = feat[feat.index >= pd.Timestamp(start_date)]

    missing_required = [c for c in BASELINE_FEATURE_COLUMNS if c not in feat.columns]
    if missing_required:
        return feat.iloc[0:0], missing_required

    valid_mask = feat[list(BASELINE_FEATURE_COLUMNS)].notna().all(axis=1)
    return feat.loc[valid_mask], []


def _write_runner_readme(outputs_dir: Path) -> Path:
    readme_path = outputs_dir / "live_runner_readme.md"
    text = """# Live Monthly Runner (Frozen H3 Champion)

## How To Run
From project root:

```bash
python run_monthly_signal.py --dry-run
```

Readiness check:

```bash
python run_monthly_signal.py --readiness-check
python run_monthly_signal.py --readiness-check --as-of-date 2026-03-03
```

Recommended live-ops dry-run command:

```bash
python run_monthly_signal.py --dry-run --signal-month 2026-02 --allow-outside-window-dry-run
```

## What It Writes
- `outputs/live_signal.json`
- `outputs/live_trade_ticket.json`
- `outputs/live_operator_summary.txt`
- `outputs/live_next_window_status.json`
- `outputs/live_next_window_status.txt`
- `outputs/live_signal_history.csv` (append-only)
- `outputs/live_runner_log.csv` (append-only)
- `outputs/live_runner_readme.md` (this file)

## Key Flags
- `--signal-month YYYY-MM`: evaluate a specific completed month.
- `--readiness-check`: run next-window operational readiness diagnostics.
- `--as-of-date YYYY-MM-DD`: simulate calendar date for window validation and operator messaging only.
- Default execution-window behavior is strict.
- `--allow-outside-window-dry-run`: allow artifact generation outside window in dry-run mode while forcing `NO_ACTION`.

## How To Interpret INVEST/CASH
- `target_state = INVEST`: next-month target is 100% one broad-market ETF.
- `target_state = CASH`: next-month target is 0% ETF and 100% cash sweep/MMF.
- Strategy is frozen: ridge + baseline features + h3 + sign rule + 1M lag.

## Manual Trade Timing
1. Run `--readiness-check` first to verify window status and blockers.
2. Run runner at month-end/turn-of-month for the intended signal month.
3. Verify `validation_status` is `PASS`.
4. Verify `manual_execution_allowed` is `true` in trade ticket.
5. Open `outputs/live_trade_ticket.json`.
6. Manually place the order within the allowed execution window.
7. If execution is not allowed, do not trade.

## Safety Behavior
If any required safety check fails, runner sets:
- `recommended_action = NO_ACTION`
- `validation_status = FAIL`
- explicit failure reason in JSON artifacts and log CSV
"""
    readme_path.write_text(text, encoding="utf-8")
    return readme_path


def _write_next_window_status_text(outputs_dir: Path, payload: dict[str, Any]) -> Path:
    status_txt_path = outputs_dir / "live_next_window_status.txt"
    lines = [
        f"Run timestamp: {payload['run_timestamp']}",
        f"As-of date: {payload['as_of_date']} (simulated={payload['as_of_date_simulated']})",
        f"Latest available data month: {payload['latest_available_data_month']}",
        f"Latest evaluable signal month: {payload['latest_evaluable_signal_month']}",
        f"Next signal month target: {payload['next_signal_month']}",
        f"Next execution month: {payload['next_execution_month']}",
        (
            "Next execution window (America/New_York): "
            f"{payload['next_execution_window']['start_date']} to {payload['next_execution_window']['end_date']} "
            f"[{payload['next_execution_window']['position_today']}]"
        ),
        f"Manual execution allowed today: {payload['manual_execution_would_be_allowed_today']}",
    ]
    if payload["blocker"]:
        lines.append(f"Blocker: {payload['blocker']}")

    checklist = payload["operator_checklist"]
    lines.extend(
        [
            "",
            "Next-Window Operator Checklist:",
            f"- Data refresh timing: {checklist['data_refresh_timing']}",
            f"- Earliest valid run date: {checklist['earliest_valid_run_date']}",
            f"- Last allowed execution date: {checklist['last_allowed_execution_date']}",
            f"- Current data sufficient: {checklist['current_data_sufficient']}",
            f"- Data sufficiency detail: {checklist['data_sufficiency_detail']}",
        ]
    )
    status_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status_txt_path


def _write_operator_summary(
    outputs_dir: Path,
    *,
    run_timestamp: str,
    as_of_date: str,
    as_of_date_simulated: bool,
    signal_month: str,
    signal_date: str,
    execution_month: str,
    execution_window_start: str,
    execution_window_end: str,
    target_state: str,
    y_pred_h3: float | None,
    recommended_action: str,
    manual_execution_allowed: bool,
    manual_execution_block_reason: str,
    validation_status: str,
) -> Path:
    summary_path = outputs_dir / "live_operator_summary.txt"
    pred_text = "N/A" if y_pred_h3 is None else f"{y_pred_h3:.8f}"
    allowed_text = "YES" if manual_execution_allowed else "NO"

    lines = [
        f"Run timestamp: {run_timestamp}",
        f"As-of date: {as_of_date} (simulated={as_of_date_simulated})",
        f"Evaluated signal month: {signal_month}",
        f"Signal date used: {signal_date}",
        f"Execution month: {execution_month}",
        f"Allowed execution window (America/New_York): {execution_window_start} to {execution_window_end}",
        f"Target state: {target_state}",
        f"y_pred_h3: {pred_text}",
        f"Recommended action: {recommended_action}",
        f"Manual execution currently allowed: {allowed_text}",
        f"Validation status: {validation_status}",
    ]

    if not manual_execution_allowed and manual_execution_block_reason:
        lines.append(f"Execution not allowed reason: {manual_execution_block_reason}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _ensure_champion_spec_artifact(outputs_dir: Path) -> Path:
    spec_path = outputs_dir / "h3_champion_spec.json"
    payload = {
        "model_version": MODEL_VERSION,
        "champion_spec": CHAMPION_SPEC,
    }
    if not spec_path.exists():
        spec_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return spec_path


def _refresh_data_pipeline(cfg: dict[str, Any]) -> None:
    run_data(cfg)
    run_features(cfg)
    run_targets(cfg)


def _live_signal_from_frozen_champion(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    start_date: str,
    train_min_periods: int,
    signal_month: pd.Period | None = None,
) -> dict[str, Any]:
    start_ts = pd.Timestamp(start_date)
    feat = features.copy()
    targ = targets.copy()
    feat.index = pd.to_datetime(feat.index)
    targ.index = pd.to_datetime(targ.index)

    feat = feat[feat.index >= start_ts]
    targ = targ[targ.index >= start_ts]

    missing_cols = [c for c in BASELINE_FEATURE_COLUMNS if c not in feat.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required baseline feature columns: {missing_cols}")

    required_feat = list(BASELINE_FEATURE_COLUMNS)
    valid_inference_mask = feat[required_feat].notna().all(axis=1)
    if not valid_inference_mask.any():
        raise RuntimeError("No valid inference row found with complete baseline feature values.")

    valid_rows = feat.loc[valid_inference_mask]
    if signal_month is None:
        signal_date = pd.Timestamp(valid_rows.index.max())
    else:
        month_mask = valid_rows.index.to_period("M") == signal_month
        if not month_mask.any():
            raise RuntimeError(
                f"No valid inference row found for requested signal month {_format_month(signal_month)}."
            )
        signal_date = pd.Timestamp(valid_rows.loc[month_mask].index.max())

    x_live = feat.loc[[signal_date], required_feat].copy()

    joined = feat[required_feat].join(targ[["target_h3"]], how="inner")
    train = joined[(joined.index < signal_date) & joined["target_h3"].notna()].dropna(subset=required_feat + ["target_h3"])
    if len(train) < int(train_min_periods):
        raise RuntimeError(
            f"Insufficient train rows for frozen champion: {len(train)} < train_min_periods={int(train_min_periods)}."
        )

    model = make_model("ridge")
    model.fit(train[required_feat], train["target_h3"].astype(float))
    y_pred_h3 = float(model.predict(x_live)[0])
    target_state = "INVEST" if y_pred_h3 > 0.0 else "CASH"

    return {
        "signal_date": signal_date,
        "signal_month": signal_date.to_period("M"),
        "y_pred_h3": y_pred_h3,
        "target_state": target_state,
        "train_rows": int(len(train)),
        "train_start": pd.Timestamp(train.index.min()),
        "train_end": pd.Timestamp(train.index.max()),
    }


def run_readiness_check(
    project_root: Path,
    config_path: str = "config.yaml",
    refresh_data: bool = False,
    max_data_age_days: int = 62,
    execution_window_business_days: int = 3,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    cfg = _load_config(project_root / config_path)
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    readme_path = _write_runner_readme(outputs_dir)

    run_ts = datetime.now(UTC)
    run_ts_iso = run_ts.isoformat()
    run_date, as_of_date_simulated = _resolve_run_date(as_of_date)

    refresh_status = "skipped"
    refresh_detail = "refresh skipped; using current available snapshot"
    if refresh_data:
        try:
            _refresh_data_pipeline(cfg)
            refresh_status = "ok"
            refresh_detail = "data/features/targets refresh completed"
        except Exception as exc:  # pragma: no cover - runtime dependency path
            refresh_status = "failed"
            refresh_detail = f"refresh failed: {exc}"

    feature_path = project_root / cfg["paths"]["features"]
    raw_path = project_root / cfg["paths"]["raw_data"]

    if not feature_path.exists():
        raise RuntimeError(f"Missing features file: {feature_path}")
    if not raw_path.exists():
        raise RuntimeError(f"Missing raw data file: {raw_path}")

    features = pd.read_csv(feature_path, parse_dates=["date"], index_col="date")
    raw = pd.read_csv(raw_path, parse_dates=["date"], index_col="date")

    _ensure_champion_spec_artifact(outputs_dir)

    latest_feature_month = pd.Timestamp(features.index.max()).to_period("M")
    latest_raw_month = pd.Timestamp(raw.index.max()).to_period("M")
    data_timestamp = pd.Timestamp(raw.index.max())
    data_age_days = int((run_date - data_timestamp).days)
    data_fresh_enough = bool(data_age_days <= int(max_data_age_days))

    valid_rows, missing_required = _valid_inference_rows(features, str(CHAMPION_SPEC["start_date"]))
    has_valid_rows = not valid_rows.empty

    if has_valid_rows:
        latest_evaluable_signal_date = pd.Timestamp(valid_rows.index.max())
        latest_evaluable_signal_month = latest_evaluable_signal_date.to_period("M")
        latest_exec_start, latest_exec_end = _execution_window_from_signal_date(
            latest_evaluable_signal_date,
            execution_window_business_days=int(execution_window_business_days),
        )
        latest_window_position = _window_position(run_date, latest_exec_start, latest_exec_end)
    else:
        latest_evaluable_signal_date = pd.NaT
        latest_evaluable_signal_month = None
        latest_exec_start = pd.NaT
        latest_exec_end = pd.NaT
        latest_window_position = "unknown"

    if latest_evaluable_signal_month is None:
        next_signal_month = latest_feature_month
    elif latest_window_position == "after":
        next_signal_month = latest_evaluable_signal_month + 1
    else:
        next_signal_month = latest_evaluable_signal_month

    if has_valid_rows:
        next_month_mask = valid_rows.index.to_period("M") == next_signal_month
        next_signal_month_data_available = bool(next_month_mask.any())
        if next_signal_month_data_available:
            next_signal_date = pd.Timestamp(valid_rows.loc[next_month_mask].index.max())
        else:
            next_signal_date = _month_end_timestamp(next_signal_month)
    else:
        next_signal_month_data_available = False
        next_signal_date = _month_end_timestamp(next_signal_month)

    next_exec_start, next_exec_end = _execution_window_from_signal_date(
        next_signal_date,
        execution_window_business_days=int(execution_window_business_days),
    )
    next_execution_month = next_exec_start.to_period("M")
    next_window_position = _window_position(run_date, next_exec_start, next_exec_end)

    blockers: list[str] = []
    if refresh_status == "failed":
        blockers.append(refresh_detail)
    if missing_required:
        blockers.append(f"Missing required baseline feature columns: {missing_required}")
    if not has_valid_rows:
        blockers.append("No valid inference rows with complete baseline feature values from champion start date.")
    if latest_raw_month != latest_feature_month:
        blockers.append(
            (
                "Latest raw and feature data months are misaligned "
                f"(raw={_format_month(latest_raw_month)}, features={_format_month(latest_feature_month)})."
            )
        )
    if not data_fresh_enough:
        blockers.append(
            (
                "Data freshness exceeds threshold "
                f"(age_days={data_age_days}, max_allowed={int(max_data_age_days)})."
            )
        )
    if not next_signal_month_data_available:
        blockers.append(
            (
                f"Data for next signal month {_format_month(next_signal_month)} is not available "
                f"(latest evaluable month={_format_month(latest_evaluable_signal_month) if latest_evaluable_signal_month else 'N/A'})."
            )
        )
    if next_window_position == "before":
        blockers.append(
            (
                "Execution window has not opened yet "
                f"(opens {next_exec_start.date().isoformat()})."
            )
        )
    elif next_window_position == "after":
        blockers.append(
            (
                "Execution window has already passed "
                f"(closed {next_exec_end.date().isoformat()})."
            )
        )

    manual_execution_would_be_allowed_today = bool(
        next_window_position == "inside"
        and next_signal_month_data_available
        and data_fresh_enough
        and len(missing_required) == 0
        and latest_raw_month == latest_feature_month
        and refresh_status != "failed"
    )

    blocker = "" if manual_execution_would_be_allowed_today else "; ".join(blockers)

    current_data_sufficient = bool(
        next_signal_month_data_available
        and len(missing_required) == 0
        and latest_raw_month == latest_feature_month
    )
    if current_data_sufficient:
        data_sufficiency_detail = (
            f"Current data can support next signal month {_format_month(next_signal_month)}."
        )
        data_refresh_timing = (
            f"Refresh is recommended before {next_exec_start.date().isoformat()} but not strictly required for readiness."
        )
    else:
        data_sufficiency_detail = (
            f"Current data is missing for next signal month {_format_month(next_signal_month)} "
            f"(latest available data month is {_format_month(latest_feature_month)})."
        )
        data_refresh_timing = (
            f"Refresh data/features/targets before {next_exec_start.date().isoformat()} "
            f"to enable signal month {_format_month(next_signal_month)}."
        )

    status_payload = {
        "run_timestamp": run_ts_iso,
        "as_of_date": run_date.date().isoformat(),
        "as_of_date_simulated": bool(as_of_date_simulated),
        "refresh_status": refresh_status,
        "refresh_detail": refresh_detail,
        "model_version": MODEL_VERSION,
        "latest_available_data_month": _format_month(latest_feature_month),
        "latest_raw_data_month": _format_month(latest_raw_month),
        "latest_evaluable_signal_month": _format_month(latest_evaluable_signal_month) if latest_evaluable_signal_month else None,
        "latest_evaluable_signal_date": (
            latest_evaluable_signal_date.date().isoformat() if pd.notna(latest_evaluable_signal_date) else None
        ),
        "latest_evaluable_execution_window": {
            "start_date": latest_exec_start.date().isoformat() if pd.notna(latest_exec_start) else None,
            "end_date": latest_exec_end.date().isoformat() if pd.notna(latest_exec_end) else None,
            "position_today": latest_window_position,
        },
        "next_signal_month": _format_month(next_signal_month),
        "next_signal_month_data_available": bool(next_signal_month_data_available),
        "next_signal_date": next_signal_date.date().isoformat(),
        "next_execution_month": _format_month(next_execution_month),
        "next_execution_window": {
            "start_date": next_exec_start.date().isoformat(),
            "end_date": next_exec_end.date().isoformat(),
            "position_today": next_window_position,
            "timezone": "America/New_York",
        },
        "manual_execution_would_be_allowed_today": bool(manual_execution_would_be_allowed_today),
        "blocker": blocker,
        "data_freshness": {
            "data_timestamp": data_timestamp.date().isoformat(),
            "age_days": int(data_age_days),
            "max_data_age_days": int(max_data_age_days),
            "fresh_enough": bool(data_fresh_enough),
        },
        "operator_checklist": {
            "data_refresh_timing": data_refresh_timing,
            "earliest_valid_run_date": next_exec_start.date().isoformat(),
            "last_allowed_execution_date": next_exec_end.date().isoformat(),
            "current_data_sufficient": bool(current_data_sufficient),
            "data_sufficiency_detail": data_sufficiency_detail,
        },
    }

    status_json_path = outputs_dir / "live_next_window_status.json"
    status_json_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
    status_txt_path = _write_next_window_status_text(outputs_dir, status_payload)

    return {
        "live_next_window_status_json": str(status_json_path),
        "live_next_window_status_txt": str(status_txt_path),
        "live_runner_readme_path": str(readme_path),
        "manual_execution_would_be_allowed_today": bool(manual_execution_would_be_allowed_today),
        "blocker": blocker,
    }


def run_monthly_signal(
    project_root: Path,
    config_path: str = "config.yaml",
    ticker: str = "SPY",
    refresh_data: bool = False,
    dry_run: bool = True,
    max_data_age_days: int = 62,
    execution_window_business_days: int = 3,
    signal_month: str | None = None,
    allow_outside_window_dry_run: bool = False,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    cfg = _load_config(project_root / config_path)
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    readme_path = _write_runner_readme(outputs_dir)

    run_ts = datetime.now(UTC)
    run_ts_iso = run_ts.isoformat()
    run_date, as_of_date_simulated = _resolve_run_date(as_of_date)

    requested_signal_month = _parse_signal_month(signal_month)

    safety_checks: list[SafetyCheck] = []
    refresh_error = ""
    if refresh_data:
        try:
            _refresh_data_pipeline(cfg)
            safety_checks.append(SafetyCheck("data_refresh_step", True, "data/features/targets refresh completed"))
        except Exception as exc:  # pragma: no cover - runtime dependency path
            refresh_error = str(exc)
            safety_checks.append(SafetyCheck("data_refresh_step", False, f"refresh failed: {refresh_error}"))
    else:
        safety_checks.append(SafetyCheck("data_refresh_step", True, "refresh skipped; using current available snapshot"))

    if as_of_date_simulated:
        safety_checks.append(
            SafetyCheck(
                "as_of_date_simulation",
                True,
                f"using simulated as_of_date={run_date.date().isoformat()} for validation and messaging",
            )
        )

    feature_path = project_root / cfg["paths"]["features"]
    target_path = project_root / cfg["paths"]["targets"]
    raw_path = project_root / cfg["paths"]["raw_data"]

    if not feature_path.exists():
        raise RuntimeError(f"Missing features file: {feature_path}")
    if not target_path.exists():
        raise RuntimeError(f"Missing targets file: {target_path}")
    if not raw_path.exists():
        raise RuntimeError(f"Missing raw data file: {raw_path}")

    features = pd.read_csv(feature_path, parse_dates=["date"], index_col="date")
    targets = pd.read_csv(target_path, parse_dates=["date"], index_col="date")
    raw = pd.read_csv(raw_path, parse_dates=["date"], index_col="date")

    spec_path = _ensure_champion_spec_artifact(outputs_dir)
    safety_checks.append(SafetyCheck("model_version_exists", spec_path.exists(), f"spec artifact: {spec_path.name}"))

    missing_required = [c for c in BASELINE_FEATURE_COLUMNS if c not in features.columns]
    safety_checks.append(
        SafetyCheck(
            "required_columns_present",
            len(missing_required) == 0,
            "all required baseline columns present" if not missing_required else f"missing: {missing_required}",
        )
    )

    latest_feature_month = pd.Timestamp(features.index.max()).to_period("M")
    latest_raw_month = pd.Timestamp(raw.index.max()).to_period("M")
    intended_signal_month = requested_signal_month if requested_signal_month is not None else latest_feature_month
    data_matches_signal_month = latest_feature_month == intended_signal_month

    safety_checks.append(
        SafetyCheck(
            "signal_month_data_alignment",
            bool(data_matches_signal_month),
            (
                f"latest_feature_month={_format_month(latest_feature_month)} "
                f"intended_signal_month={_format_month(intended_signal_month)} "
                f"match={bool(data_matches_signal_month)}"
            ),
        )
    )

    safety_checks.append(
        SafetyCheck(
            "raw_vs_feature_latest_month_alignment",
            bool(latest_raw_month == latest_feature_month),
            (
                f"latest_raw_month={_format_month(latest_raw_month)} "
                f"latest_feature_month={_format_month(latest_feature_month)}"
            ),
        )
    )

    signal_generation_ok = True
    signal_generation_error = ""
    signal_payload: dict[str, Any]

    try:
        signal_payload = _live_signal_from_frozen_champion(
            features=features,
            targets=targets,
            start_date=str(CHAMPION_SPEC["start_date"]),
            train_min_periods=int(CHAMPION_SPEC["train_min_periods"]),
            signal_month=requested_signal_month,
        )
        signal_date = pd.Timestamp(signal_payload["signal_date"])
        signal_month_period = pd.Period(signal_payload["signal_month"], freq="M")
        target_state = str(signal_payload["target_state"])
        y_pred_h3: float | None = float(signal_payload["y_pred_h3"])
    except Exception as exc:
        signal_generation_ok = False
        signal_generation_error = str(exc)
        signal_month_period = intended_signal_month
        signal_date = _month_end_timestamp(signal_month_period)
        target_state = "UNKNOWN"
        y_pred_h3 = None
        signal_payload = {
            "train_rows": 0,
            "train_start": pd.NaT,
            "train_end": pd.NaT,
        }

    safety_checks.append(
        SafetyCheck(
            "signal_generation",
            bool(signal_generation_ok),
            "signal generated successfully" if signal_generation_ok else f"signal generation failed: {signal_generation_error}",
        )
    )

    if signal_generation_ok and signal_date in features.index:
        latest_feature_row = features.loc[[signal_date], list(BASELINE_FEATURE_COLUMNS)]
        no_missing_live_inputs = bool(latest_feature_row.notna().all(axis=1).iloc[0])
        inputs_detail = f"signal_date={signal_date.date()} feature row complete={no_missing_live_inputs}"
    else:
        no_missing_live_inputs = False
        inputs_detail = "signal_date row unavailable because signal generation failed"
    safety_checks.append(SafetyCheck("no_missing_current_inputs", no_missing_live_inputs, inputs_detail))

    data_timestamp = pd.Timestamp(raw.index.max())
    data_age_days = int((run_date - data_timestamp).days)
    freshness_ok = data_age_days <= int(max_data_age_days)
    safety_checks.append(
        SafetyCheck(
            "data_freshness_check",
            bool(freshness_ok),
            (
                f"data_timestamp={data_timestamp.date()} age_days={data_age_days} max_allowed={int(max_data_age_days)} "
                f"latest_data_month={_format_month(latest_feature_month)} intended_signal_month={_format_month(intended_signal_month)}"
            ),
        )
    )

    signal_ok = target_state in {"INVEST", "CASH"}
    safety_checks.append(SafetyCheck("signal_enum_check", signal_ok, f"target_state={target_state}"))

    exec_start, exec_end = _execution_window_from_signal_date(
        signal_date,
        execution_window_business_days=int(execution_window_business_days),
    )
    execution_month = exec_start.to_period("M")
    window_position = _window_position(run_date, exec_start, exec_end)
    in_window_now = window_position == "inside"
    outside_window_dry_run_applied = bool(
        allow_outside_window_dry_run and dry_run and not in_window_now
    )

    if in_window_now:
        safety_checks.append(
            SafetyCheck(
                "execution_window_check",
                True,
                f"run_date={run_date.date()} inside allowed_window={exec_start.date()}..{exec_end.date()}",
            )
        )
    elif outside_window_dry_run_applied:
        safety_checks.append(
            SafetyCheck(
                "execution_window_check",
                True,
                (
                    f"run_date={run_date.date()} outside allowed_window={exec_start.date()}..{exec_end.date()} "
                    "but --allow-outside-window-dry-run enabled; NO_ACTION enforced"
                ),
            )
        )
    else:
        safety_checks.append(
            SafetyCheck(
                "execution_window_check",
                False,
                f"run_date={run_date.date()} outside allowed_window={exec_start.date()}..{exec_end.date()}",
            )
        )

    constraints_encoded = True
    safety_checks.append(
        SafetyCheck(
            "constraints_encoded",
            constraints_encoded,
            "no_leverage=True, no_short=True, one_etf_only=True encoded in ticket",
        )
    )

    failed_checks = [c for c in safety_checks if not c.passed]
    validation_status = "PASS" if not failed_checks else "FAIL"
    failure_reason = "; ".join([f"{c.name}: {c.detail}" for c in failed_checks]) if failed_checks else ""

    history_path = outputs_dir / "live_signal_history.csv"
    prev_target_state = ""
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        if not history_df.empty and "target_state" in history_df.columns:
            prev_target_state = str(history_df.iloc[-1]["target_state"])

    base_action = "NO_ACTION"
    if target_state == "INVEST":
        base_action = "HOLD" if prev_target_state == "INVEST" else "BUY"
    elif target_state == "CASH":
        base_action = "HOLD" if prev_target_state == "CASH" else "SELL"

    recommended_action = base_action if validation_status == "PASS" else "NO_ACTION"
    if outside_window_dry_run_applied:
        recommended_action = "NO_ACTION"

    manual_execution_allowed = bool(validation_status == "PASS" and in_window_now and not dry_run and not as_of_date_simulated)
    if manual_execution_allowed:
        manual_execution_block_reason = ""
    elif as_of_date_simulated:
        manual_execution_block_reason = "As-of-date simulation mode is for operational testing only; manual execution disabled."
    elif outside_window_dry_run_applied:
        manual_execution_block_reason = (
            "Outside execution window with --allow-outside-window-dry-run: artifacts produced, NO_ACTION enforced."
        )
    elif dry_run and validation_status == "PASS" and in_window_now:
        manual_execution_block_reason = "Dry-run mode enabled; manual execution disabled for this run."
    elif validation_status != "PASS":
        manual_execution_block_reason = failure_reason
    else:
        manual_execution_block_reason = "Execution not allowed by current run constraints."

    signal_artifact = {
        "run_timestamp": run_ts_iso,
        "as_of_date": run_date.date().isoformat(),
        "as_of_date_simulated": bool(as_of_date_simulated),
        "dry_run": bool(dry_run),
        "requested_signal_month": _format_month(requested_signal_month) if requested_signal_month else None,
        "signal_month": _format_month(signal_month_period),
        "execution_month": _format_month(execution_month),
        "signal_date": signal_date.date().isoformat(),
        "y_pred_h3": y_pred_h3,
        "target_state": target_state,
        "model_version": MODEL_VERSION,
        "model_name": CHAMPION_SPEC["model"],
        "horizon": CHAMPION_SPEC["horizon"],
        "data_timestamp": data_timestamp.date().isoformat(),
        "latest_data_month": _format_month(latest_feature_month),
        "data_matches_signal_month": bool(data_matches_signal_month),
        "train_rows": int(signal_payload["train_rows"]),
        "train_start": (
            pd.Timestamp(signal_payload["train_start"]).date().isoformat()
            if pd.notna(signal_payload["train_start"])
            else None
        ),
        "train_end": (
            pd.Timestamp(signal_payload["train_end"]).date().isoformat()
            if pd.notna(signal_payload["train_end"])
            else None
        ),
        "validation_status": validation_status,
        "failure_reason": failure_reason,
        "manual_execution_allowed": manual_execution_allowed,
        "manual_execution_block_reason": manual_execution_block_reason,
        "safety_checks": [c.as_dict() for c in safety_checks],
    }

    trade_ticket = {
        "run_timestamp": run_ts_iso,
        "as_of_date": run_date.date().isoformat(),
        "as_of_date_simulated": bool(as_of_date_simulated),
        "dry_run": bool(dry_run),
        "signal_month": _format_month(signal_month_period),
        "execution_month": _format_month(execution_month),
        "recommended_action": recommended_action,
        "action": recommended_action,
        "ticker": ticker,
        "target_state": target_state,
        "manual_execution_allowed": manual_execution_allowed,
        "manual_execution_block_reason": manual_execution_block_reason,
        "reasoning_stub": (
            "Frozen h3 champion monthly decision. Human review required before manual broker order."
            if validation_status == "PASS" and not outside_window_dry_run_applied and not as_of_date_simulated
            else "Trade not currently allowed. Review validation and execution window status."
        ),
        "allowed_execution_window": {
            "start_date": exec_start.date().isoformat(),
            "end_date": exec_end.date().isoformat(),
            "position_today": window_position,
            "timezone": "America/New_York",
        },
        "constraints": {
            "no_leverage": True,
            "no_short": True,
            "one_etf_only": True,
            "fractional_preferred": True,
            "whole_shares_with_residual_cash_if_no_fractional": True,
        },
        "notes": [
            "Manual execution only; no broker API integration in this step.",
            "If validation_status != PASS, recommended_action must remain NO_ACTION.",
            "If outside execution window, recommended_action must remain NO_ACTION unless running live in-window.",
            f"validation_status={validation_status}",
        ],
        "validation_status": validation_status,
        "failure_reason": failure_reason,
    }

    live_signal_path = outputs_dir / "live_signal.json"
    live_ticket_path = outputs_dir / "live_trade_ticket.json"
    live_runner_log_path = outputs_dir / "live_runner_log.csv"

    live_signal_path.write_text(json.dumps(signal_artifact, indent=2), encoding="utf-8")
    live_ticket_path.write_text(json.dumps(trade_ticket, indent=2), encoding="utf-8")

    operator_summary_path = _write_operator_summary(
        outputs_dir,
        run_timestamp=run_ts_iso,
        as_of_date=run_date.date().isoformat(),
        as_of_date_simulated=bool(as_of_date_simulated),
        signal_month=signal_artifact["signal_month"],
        signal_date=signal_artifact["signal_date"],
        execution_month=signal_artifact["execution_month"],
        execution_window_start=trade_ticket["allowed_execution_window"]["start_date"],
        execution_window_end=trade_ticket["allowed_execution_window"]["end_date"],
        target_state=target_state,
        y_pred_h3=y_pred_h3,
        recommended_action=recommended_action,
        manual_execution_allowed=manual_execution_allowed,
        manual_execution_block_reason=manual_execution_block_reason,
        validation_status=validation_status,
    )

    history_row = {
        "run_timestamp": run_ts_iso,
        "as_of_date": run_date.date().isoformat(),
        "as_of_date_simulated": bool(as_of_date_simulated),
        "signal_month": signal_artifact["signal_month"],
        "execution_month": signal_artifact["execution_month"],
        "signal_date": signal_artifact["signal_date"],
        "y_pred_h3": y_pred_h3,
        "target_state": target_state,
        "recommended_action": recommended_action,
        "validation_status": validation_status,
        "failure_reason": failure_reason,
        "model_version": MODEL_VERSION,
        "data_timestamp": signal_artifact["data_timestamp"],
        "latest_data_month": signal_artifact["latest_data_month"],
        "data_matches_signal_month": bool(signal_artifact["data_matches_signal_month"]),
        "execution_window_start": exec_start.date().isoformat(),
        "execution_window_end": exec_end.date().isoformat(),
        "ticker": ticker,
        "dry_run": bool(dry_run),
        "manual_execution_allowed": manual_execution_allowed,
    }
    _append_csv_row(history_path, history_row)

    log_row = {
        "run_timestamp": run_ts_iso,
        "as_of_date": run_date.date().isoformat(),
        "as_of_date_simulated": bool(as_of_date_simulated),
        "signal_month": signal_artifact["signal_month"],
        "signal_date": signal_artifact["signal_date"],
        "data_timestamp": signal_artifact["data_timestamp"],
        "target_state": target_state,
        "recommended_action": recommended_action,
        "validation_status": validation_status,
        "failure_reason": failure_reason,
        "model_version": MODEL_VERSION,
        "manual_execution_allowed": manual_execution_allowed,
        "manual_execution_block_reason": manual_execution_block_reason,
    }
    _append_csv_row(live_runner_log_path, log_row)

    return {
        "live_signal_path": str(live_signal_path),
        "live_trade_ticket_path": str(live_ticket_path),
        "live_signal_history_path": str(history_path),
        "live_runner_log_path": str(live_runner_log_path),
        "live_runner_readme_path": str(readme_path),
        "live_operator_summary_path": str(operator_summary_path),
        "validation_status": validation_status,
        "action": recommended_action,
        "failure_reason": failure_reason,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semi-automated monthly runner for frozen h3 champion strategy.")
    parser.add_argument("--project-root", type=str, default=".", help="Project root path.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config path.")
    parser.add_argument("--ticker", type=str, default="SPY", help="Single ETF ticker for trade ticket.")
    parser.add_argument("--refresh-data", action="store_true", help="Refresh data/features/targets before signal run.")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no broker execution).")
    parser.add_argument("--max-data-age-days", type=int, default=62, help="Data freshness threshold.")
    parser.add_argument(
        "--execution-window-business-days",
        type=int,
        default=3,
        help="Allowed business-day window length starting first business day of next month.",
    )
    parser.add_argument(
        "--signal-month",
        type=str,
        default=None,
        help="Signal month in YYYY-MM format (completed month to evaluate).",
    )
    parser.add_argument(
        "--allow-outside-window-dry-run",
        action="store_true",
        help="If set with --dry-run, allow artifact generation outside execution window while forcing NO_ACTION.",
    )
    parser.add_argument(
        "--readiness-check",
        action="store_true",
        help="Run next-window readiness diagnostics and write live_next_window_status artifacts.",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Simulated date (YYYY-MM-DD) for window validation and operator messaging only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if bool(args.readiness_check):
        result = run_readiness_check(
            project_root=Path(args.project_root).resolve(),
            config_path=str(args.config),
            refresh_data=bool(args.refresh_data),
            max_data_age_days=int(args.max_data_age_days),
            execution_window_business_days=int(args.execution_window_business_days),
            as_of_date=str(args.as_of_date) if args.as_of_date is not None else None,
        )
        print("[live-readiness] wrote artifacts:")
        print(f" - live_next_window_status.json: {result['live_next_window_status_json']}")
        print(f" - live_next_window_status.txt: {result['live_next_window_status_txt']}")
        print(f" - live_runner_readme: {result['live_runner_readme_path']}")
        print(
            "[live-readiness] "
            f"manual_execution_would_be_allowed_today={result['manual_execution_would_be_allowed_today']}"
        )
        if result["blocker"]:
            print(f"[live-readiness] blocker={result['blocker']}")
        return

    result = run_monthly_signal(
        project_root=Path(args.project_root).resolve(),
        config_path=str(args.config),
        ticker=str(args.ticker),
        refresh_data=bool(args.refresh_data),
        dry_run=bool(args.dry_run),
        max_data_age_days=int(args.max_data_age_days),
        execution_window_business_days=int(args.execution_window_business_days),
        signal_month=str(args.signal_month) if args.signal_month is not None else None,
        allow_outside_window_dry_run=bool(args.allow_outside_window_dry_run),
        as_of_date=str(args.as_of_date) if args.as_of_date is not None else None,
    )
    print("[live-runner] wrote artifacts:")
    print(f" - live_signal: {result['live_signal_path']}")
    print(f" - live_trade_ticket: {result['live_trade_ticket_path']}")
    print(f" - live_signal_history: {result['live_signal_history_path']}")
    print(f" - live_runner_log: {result['live_runner_log_path']}")
    print(f" - live_operator_summary: {result['live_operator_summary_path']}")
    print(f" - live_runner_readme: {result['live_runner_readme_path']}")
    print(f"[live-runner] validation_status={result['validation_status']} action={result['action']}")
    if result["failure_reason"]:
        print(f"[live-runner] failure_reason={result['failure_reason']}")


if __name__ == "__main__":
    main()
