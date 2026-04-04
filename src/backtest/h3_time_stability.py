from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PeriodSpec:
    label: str
    start: str
    end: str | None


SUBPERIODS: tuple[PeriodSpec, ...] = (
    PeriodSpec(label="2000_2007", start="2000-01-01", end="2007-12-31"),
    PeriodSpec(label="2008_2012", start="2008-01-01", end="2012-12-31"),
    PeriodSpec(label="2013_2019", start="2013-01-01", end="2019-12-31"),
    PeriodSpec(label="2020_latest", start="2020-01-01", end=None),
)

SUBPERIODS_FULLHISTORY: tuple[PeriodSpec, ...] = (
    PeriodSpec(label="1986_1999", start="1986-11-30", end="1999-12-31"),
    PeriodSpec(label="2000_2007", start="2000-01-01", end="2007-12-31"),
    PeriodSpec(label="2008_2012", start="2008-01-01", end="2012-12-31"),
    PeriodSpec(label="2013_2019", start="2013-01-01", end="2019-12-31"),
    PeriodSpec(label="2020_latest", start="2020-01-01", end=None),
)


def _perf_stats(returns: pd.Series) -> dict[str, float]:
    rets = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
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


def _run_baseline_backtest(project_root: Path, backtest_start: str | None = "2000-01-01") -> Path:
    cmd = [
        sys.executable,
        "-m",
        "src.cli",
        "backtest",
        "--model",
        "ridge",
    ]
    if backtest_start is not None:
        cmd.extend(["--start", str(backtest_start)])

    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, check=False)

    log_out = project_root / "outputs" / "h3_time_stability_backtest.log"
    log_out.write_text((result.stdout or "") + ("\n" + result.stderr if result.stderr else ""), encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"Backtest failed for h3 time-stability diagnostics (returncode={result.returncode}). "
            f"See {log_out}."
        )
    return log_out


def _load_h3_baseline_strategy(project_root: Path) -> pd.DataFrame:
    path = project_root / "outputs" / "strategy_h3_baseline.csv"
    if not path.exists():
        raise RuntimeError(f"Expected baseline strategy file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Baseline strategy file is empty: {path}")

    required_cols = ["date", "strat_ret", "bh_ret", "turnover", "weight"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in {path.name}: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def _equity_from_returns(returns: pd.Series) -> float:
    rets = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if rets.empty:
        return float("nan")
    return float((1.0 + rets).cumprod().iloc[-1])


def _summarize_subperiod(strategy_df: pd.DataFrame, spec: PeriodSpec) -> dict[str, float | int | str]:
    start_ts = pd.Timestamp(spec.start)
    if spec.end is None:
        if strategy_df.empty:
            end_ts = start_ts
        else:
            end_ts = pd.to_datetime(strategy_df["date"]).max()
        end_label = pd.Timestamp(end_ts).date().isoformat()
    else:
        end_ts = pd.Timestamp(spec.end)
        end_label = spec.end
    window = strategy_df[(strategy_df["date"] >= start_ts) & (strategy_df["date"] <= end_ts)].copy()

    strat_rets = pd.to_numeric(window["strat_ret"], errors="coerce").dropna()
    bh_rets = pd.to_numeric(window["bh_ret"], errors="coerce").dropna()
    turnover = pd.to_numeric(window["turnover"], errors="coerce")
    weight = pd.to_numeric(window["weight"], errors="coerce")

    strat_stats = _perf_stats(strat_rets)
    bh_stats = _perf_stats(bh_rets)

    observed_start = pd.to_datetime(window["date"]).min().date().isoformat() if not window.empty else ""
    observed_end = pd.to_datetime(window["date"]).max().date().isoformat() if not window.empty else ""

    return {
        "subperiod": spec.label,
        "start_date": spec.start,
        "end_date": end_label,
        "observed_start_date": observed_start,
        "observed_end_date": observed_end,
        "months": int(len(strat_rets)),
        "final_equity": _equity_from_returns(strat_rets),
        "CAGR": strat_stats["cagr"],
        "vol": strat_stats["vol"],
        "Sharpe": strat_stats["sharpe"],
        "max_drawdown": strat_stats["max_dd"],
        "avg_turnover": float(turnover.mean()) if not turnover.empty else float("nan"),
        "total_turnover": float(turnover.sum()) if not turnover.empty else float("nan"),
        "hit_rate": float((strat_rets > 0.0).mean()) if not strat_rets.empty else float("nan"),
        "invested_fraction": float(weight.mean()) if not weight.empty else float("nan"),
        "bh_final_equity": _equity_from_returns(bh_rets),
        "bh_CAGR": bh_stats["cagr"],
        "bh_vol": bh_stats["vol"],
        "bh_Sharpe": bh_stats["sharpe"],
        "bh_max_drawdown": bh_stats["max_dd"],
    }


def _build_subperiod_summary(
    strategy_df: pd.DataFrame,
    subperiods: tuple[PeriodSpec, ...] = SUBPERIODS,
) -> pd.DataFrame:
    rows = [_summarize_subperiod(strategy_df=strategy_df, spec=spec) for spec in subperiods]
    cols = [
        "subperiod",
        "start_date",
        "end_date",
        "observed_start_date",
        "observed_end_date",
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
        "bh_final_equity",
        "bh_CAGR",
        "bh_vol",
        "bh_Sharpe",
        "bh_max_drawdown",
    ]
    return pd.DataFrame(rows)[cols]


def _build_rolling_summary(strategy_df: pd.DataFrame, window_months: int = 36) -> pd.DataFrame:
    if window_months <= 0:
        raise ValueError("window_months must be a positive integer.")

    cols = [
        "window_start_date",
        "window_end_date",
        "months",
        "rolling_36m_CAGR",
        "rolling_36m_Sharpe",
        "rolling_36m_hit_rate",
        "rolling_36m_invested_fraction",
    ]
    if strategy_df.empty or len(strategy_df) < window_months:
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, float | int | str]] = []
    for end_idx in range(window_months - 1, len(strategy_df)):
        window = strategy_df.iloc[end_idx - window_months + 1 : end_idx + 1].copy()
        strat_rets = pd.to_numeric(window["strat_ret"], errors="coerce").dropna()
        weight = pd.to_numeric(window["weight"], errors="coerce")
        if len(strat_rets) < window_months:
            continue

        perf = _perf_stats(strat_rets)
        rows.append(
            {
                "window_start_date": pd.to_datetime(window["date"].iloc[0]).date().isoformat(),
                "window_end_date": pd.to_datetime(window["date"].iloc[-1]).date().isoformat(),
                "months": int(len(strat_rets)),
                "rolling_36m_CAGR": perf["cagr"],
                "rolling_36m_Sharpe": perf["sharpe"],
                "rolling_36m_hit_rate": float((strat_rets > 0.0).mean()),
                "rolling_36m_invested_fraction": float(weight.mean()) if not weight.empty else float("nan"),
            }
        )

    return pd.DataFrame(rows, columns=cols)


def build_h3_time_stability(
    project_root: Path,
    run_backtest: bool = True,
    window_months: int = 36,
    subperiods: tuple[PeriodSpec, ...] = SUBPERIODS,
    subperiod_output_name: str = "strategy_h3_subperiod_summary.csv",
    rolling_output_name: str = "strategy_h3_rolling_36m.csv",
    backtest_start: str | None = "2000-01-01",
) -> dict[str, Path]:
    outputs = project_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    if run_backtest:
        _run_baseline_backtest(project_root=project_root, backtest_start=backtest_start)

    strategy_df = _load_h3_baseline_strategy(project_root=project_root)
    subperiod_summary = _build_subperiod_summary(strategy_df=strategy_df, subperiods=subperiods)
    rolling_summary = _build_rolling_summary(strategy_df=strategy_df, window_months=window_months)

    subperiod_out = outputs / str(subperiod_output_name)
    rolling_out = outputs / str(rolling_output_name)
    subperiod_summary.to_csv(subperiod_out, index=False)
    rolling_summary.to_csv(rolling_out, index=False)

    return {
        "subperiod_summary": subperiod_out,
        "rolling_36m": rolling_out,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build h3 baseline time-stability diagnostics.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root path (contains src/ and outputs/).",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip rerunning baseline backtest and use existing outputs/strategy_h3_baseline.csv.",
    )
    parser.add_argument(
        "--window-months",
        type=int,
        default=36,
        help="Rolling window length in months (default: 36).",
    )
    parser.add_argument(
        "--subperiod-profile",
        type=str,
        default="default",
        choices=["default", "fullhistory"],
        help="Subperiod definition profile.",
    )
    parser.add_argument(
        "--subperiod-output",
        type=str,
        default="strategy_h3_subperiod_summary.csv",
        help="Output CSV filename for subperiod diagnostics (written under outputs/).",
    )
    parser.add_argument(
        "--rolling-output",
        type=str,
        default="strategy_h3_rolling_36m.csv",
        help="Output CSV filename for rolling diagnostics (written under outputs/).",
    )
    parser.add_argument(
        "--backtest-start",
        type=str,
        default="2000-01-01",
        help="Start date passed to backtest; use 'none' to omit start filter entirely.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    subperiods = SUBPERIODS if args.subperiod_profile == "default" else SUBPERIODS_FULLHISTORY
    backtest_start = None if str(args.backtest_start).strip().lower() == "none" else str(args.backtest_start)
    artifacts = build_h3_time_stability(
        project_root=project_root,
        run_backtest=not bool(args.skip_backtest),
        window_months=int(args.window_months),
        subperiods=subperiods,
        subperiod_output_name=str(args.subperiod_output),
        rolling_output_name=str(args.rolling_output),
        backtest_start=backtest_start,
    )
    print("[h3-time-stability] wrote:")
    for key, path in artifacts.items():
        print(f" - {key}: {path}")


if __name__ == "__main__":
    main()
