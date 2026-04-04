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
    PeriodSpec(label="full_history", start="1900-01-01", end=None),
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
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, check=False)

    log_out = project_root / "outputs" / "h3_return_attribution_backtest.log"
    log_out.write_text((result.stdout or "") + ("\n" + result.stderr if result.stderr else ""), encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Backtest failed for h3 attribution diagnostics (returncode={result.returncode}). See {log_out}.")
    return log_out


def _load_h3_strategy(project_root: Path) -> pd.DataFrame:
    path = project_root / "outputs" / "strategy_h3_baseline.csv"
    if not path.exists():
        raise RuntimeError(f"Expected strategy output not found: {path}")

    df = pd.read_csv(path)
    required = [
        "date",
        "weight",
        "strat_ret_gross",
        "strat_ret",
        "bh_ret",
        "cash_ret",
        "turnover",
        "cost",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in {path.name}: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def _slice_period(df: pd.DataFrame, spec: PeriodSpec) -> pd.DataFrame:
    start_ts = pd.Timestamp(spec.start)
    if spec.end is None:
        end_ts = pd.to_datetime(df["date"]).max() if not df.empty else start_ts
    else:
        end_ts = pd.Timestamp(spec.end)
    return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()


def _compute_return_attribution(window: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    if window.empty:
        return {
            "subperiod": label,
            "observed_start_date": "",
            "observed_end_date": "",
            "months": 0,
            "final_equity": float("nan"),
            "CAGR": float("nan"),
            "vol": float("nan"),
            "Sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "avg_turnover": float("nan"),
            "total_turnover": float("nan"),
            "equity_leg_return": float("nan"),
            "cash_leg_return": float("nan"),
            "cost_drag_return": float("nan"),
            "net_strategy_return": float("nan"),
            "gross_pre_cost_return": float("nan"),
            "equity_share_of_net": float("nan"),
            "cash_share_of_net": float("nan"),
            "cost_drag_share_of_net": float("nan"),
        }

    work = window.copy()
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce")
    work["bh_ret"] = pd.to_numeric(work["bh_ret"], errors="coerce")
    work["strat_ret"] = pd.to_numeric(work["strat_ret"], errors="coerce")
    work["strat_ret_gross"] = pd.to_numeric(work["strat_ret_gross"], errors="coerce")
    work["cost"] = pd.to_numeric(work["cost"], errors="coerce")
    work["turnover"] = pd.to_numeric(work["turnover"], errors="coerce")

    # Use model-consistent decomposition:
    # strat_ret = (weight*bh_ret) + (strat_ret_gross - weight*bh_ret) - cost
    work["equity_leg_ret"] = work["weight"] * work["bh_ret"]
    work["cash_leg_ret"] = work["strat_ret_gross"] - work["equity_leg_ret"]
    work["cost_leg_ret"] = -work["cost"]

    equity_curve = 1.0
    equity_pnl = 0.0
    cash_pnl = 0.0
    cost_pnl = 0.0
    for row in work.itertuples(index=False):
        eq_start = equity_curve
        eq_component = eq_start * float(row.equity_leg_ret)
        cash_component = eq_start * float(row.cash_leg_ret)
        cost_component = eq_start * float(row.cost_leg_ret)
        equity_pnl += eq_component
        cash_pnl += cash_component
        cost_pnl += cost_component
        equity_curve = eq_start * (1.0 + float(row.strat_ret))

    net_return = float(equity_curve - 1.0)
    gross_pre_cost = float(equity_pnl + cash_pnl)
    perf = _perf_stats(work["strat_ret"])
    avg_turnover = float(work["turnover"].mean())
    total_turnover = float(work["turnover"].sum())

    denom = net_return if not np.isclose(net_return, 0.0) else float("nan")
    return {
        "subperiod": label,
        "observed_start_date": pd.to_datetime(work["date"]).min().date().isoformat(),
        "observed_end_date": pd.to_datetime(work["date"]).max().date().isoformat(),
        "months": int(len(work)),
        "final_equity": float(equity_curve),
        "CAGR": perf["cagr"],
        "vol": perf["vol"],
        "Sharpe": perf["sharpe"],
        "max_drawdown": perf["max_dd"],
        "avg_turnover": avg_turnover,
        "total_turnover": total_turnover,
        "equity_leg_return": float(equity_pnl),
        "cash_leg_return": float(cash_pnl),
        "cost_drag_return": float(cost_pnl),
        "net_strategy_return": net_return,
        "gross_pre_cost_return": gross_pre_cost,
        "equity_share_of_net": float(equity_pnl / denom) if np.isfinite(denom) else float("nan"),
        "cash_share_of_net": float(cash_pnl / denom) if np.isfinite(denom) else float("nan"),
        "cost_drag_share_of_net": float(cost_pnl / denom) if np.isfinite(denom) else float("nan"),
    }


def _compute_timing_confusion(window: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    if window.empty:
        return {
            "subperiod": label,
            "observed_start_date": "",
            "observed_end_date": "",
            "months": 0,
            "tp_pos_mkt_invested": 0,
            "fn_pos_mkt_cash": 0,
            "fp_nonpos_mkt_invested": 0,
            "tn_nonpos_mkt_cash": 0,
            "true_positive_rate_pos_months": float("nan"),
            "true_negative_rate_nonpos_months": float("nan"),
            "precision_invested_months": float("nan"),
            "recall_positive_months": float("nan"),
            "avg_market_return_when_invested": float("nan"),
            "avg_market_return_when_in_cash": float("nan"),
            "avg_cash_return_when_in_cash": float("nan"),
        }

    work = window.copy()
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce")
    work["bh_ret"] = pd.to_numeric(work["bh_ret"], errors="coerce")
    work["cash_ret"] = pd.to_numeric(work["cash_ret"], errors="coerce")

    invested = work["weight"] > 0.0
    pos_market = work["bh_ret"] > 0.0

    tp = int((pos_market & invested).sum())
    fn = int((pos_market & ~invested).sum())
    fp = int((~pos_market & invested).sum())
    tn = int((~pos_market & ~invested).sum())

    pos_total = tp + fn
    nonpos_total = tn + fp
    invested_total = tp + fp

    tpr = float(tp / pos_total) if pos_total > 0 else float("nan")
    tnr = float(tn / nonpos_total) if nonpos_total > 0 else float("nan")
    precision = float(tp / invested_total) if invested_total > 0 else float("nan")

    return {
        "subperiod": label,
        "observed_start_date": pd.to_datetime(work["date"]).min().date().isoformat(),
        "observed_end_date": pd.to_datetime(work["date"]).max().date().isoformat(),
        "months": int(len(work)),
        "tp_pos_mkt_invested": tp,
        "fn_pos_mkt_cash": fn,
        "fp_nonpos_mkt_invested": fp,
        "tn_nonpos_mkt_cash": tn,
        "true_positive_rate_pos_months": tpr,
        "true_negative_rate_nonpos_months": tnr,
        "precision_invested_months": precision,
        "recall_positive_months": tpr,
        "avg_market_return_when_invested": float(work.loc[invested, "bh_ret"].mean()) if invested.any() else float("nan"),
        "avg_market_return_when_in_cash": float(work.loc[~invested, "bh_ret"].mean()) if (~invested).any() else float("nan"),
        "avg_cash_return_when_in_cash": float(work.loc[~invested, "cash_ret"].mean()) if (~invested).any() else float("nan"),
    }


def build_h3_return_attribution(
    project_root: Path,
    run_backtest: bool = True,
    backtest_start: str = "1976-10-31",
) -> dict[str, Path]:
    outputs = project_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    if run_backtest:
        _run_backtest(project_root=project_root, start_date=backtest_start)

    strat = _load_h3_strategy(project_root=project_root)

    attrib_rows: list[dict[str, float | int | str]] = []
    timing_rows: list[dict[str, float | int | str]] = []
    for spec in SUBPERIODS:
        window = _slice_period(strat, spec)
        attrib_rows.append(_compute_return_attribution(window=window, label=spec.label))
        timing_rows.append(_compute_timing_confusion(window=window, label=spec.label))

    attrib_df = pd.DataFrame(attrib_rows)
    timing_df = pd.DataFrame(timing_rows)

    timing_summary_df = timing_df[timing_df["subperiod"] == "full_history"].copy()
    timing_by_subperiod_df = timing_df[timing_df["subperiod"] != "full_history"].copy()

    attrib_out = outputs / "strategy_h3_return_attribution.csv"
    timing_summary_out = outputs / "strategy_h3_timing_confusion_summary.csv"
    timing_by_subperiod_out = outputs / "strategy_h3_timing_confusion_by_subperiod.csv"

    attrib_df.to_csv(attrib_out, index=False)
    timing_summary_df.to_csv(timing_summary_out, index=False)
    timing_by_subperiod_df.to_csv(timing_by_subperiod_out, index=False)

    return {
        "return_attribution": attrib_out,
        "timing_summary": timing_summary_out,
        "timing_by_subperiod": timing_by_subperiod_out,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Return attribution diagnostics for h3 baseline strategy.")
    parser.add_argument("--project-root", type=str, default=".", help="Project root path.")
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip rerunning baseline backtest and use existing outputs/strategy_h3_baseline.csv.",
    )
    parser.add_argument(
        "--backtest-start",
        type=str,
        default="1976-10-31",
        help="Backtest start date for max-valid full-history run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    artifacts = build_h3_return_attribution(
        project_root=project_root,
        run_backtest=not bool(args.skip_backtest),
        backtest_start=str(args.backtest_start),
    )
    print("[h3-return-attribution] wrote:")
    for key, path in artifacts.items():
        print(f" - {key}: {path}")


if __name__ == "__main__":
    main()
