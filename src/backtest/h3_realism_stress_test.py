from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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


def _build_metric_row(
    returns: pd.Series,
    turnover: pd.Series,
    scenario_type: str,
    scenario_name: str,
    scenario_value: float | str,
    bh_stats: dict[str, float],
) -> dict[str, float | int | str | bool]:
    rets = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    turnover_num = pd.to_numeric(turnover, errors="coerce")
    perf = _perf_stats(rets)
    final_equity = float((1.0 + rets).cumprod().iloc[-1]) if not rets.empty else float("nan")
    avg_turnover = float(turnover_num.mean()) if not turnover_num.empty else float("nan")
    total_turnover = float(turnover_num.sum()) if not turnover_num.empty else float("nan")
    return {
        "scenario_type": scenario_type,
        "scenario_name": scenario_name,
        "scenario_value": scenario_value,
        "months": int(len(rets)),
        "final_equity": final_equity,
        "CAGR": perf["cagr"],
        "vol": perf["vol"],
        "Sharpe": perf["sharpe"],
        "max_drawdown": perf["max_dd"],
        "avg_turnover": avg_turnover,
        "total_turnover": total_turnover,
        "bh_CAGR": float(bh_stats["cagr"]),
        "bh_vol": float(bh_stats["vol"]),
        "bh_Sharpe": float(bh_stats["sharpe"]),
        "bh_max_drawdown": float(bh_stats["max_dd"]),
        "beats_bh_sharpe": bool(perf["sharpe"] > bh_stats["sharpe"]) if np.isfinite(perf["sharpe"]) else False,
        "better_dd_than_bh": bool(perf["max_dd"] > bh_stats["max_dd"]) if np.isfinite(perf["max_dd"]) else False,
    }


def _run_baseline_backtest(project_root: Path, start_date: str) -> Path:
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
    log_out = project_root / "outputs" / "h3_realism_stress_backtest.log"
    log_out.write_text((result.stdout or "") + ("\n" + result.stderr if result.stderr else ""), encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Backtest failed for stress test (returncode={result.returncode}). See {log_out}.")
    return log_out


def _load_h3_strategy(project_root: Path) -> pd.DataFrame:
    path = project_root / "outputs" / "strategy_h3_baseline.csv"
    if not path.exists():
        raise RuntimeError(f"Missing h3 baseline strategy file: {path}")
    df = pd.read_csv(path)
    required = ["date", "weight", "turnover", "cost", "strat_ret", "bh_ret", "cash_ret"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in {path.name}: {missing}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def _infer_default_cost_rate(strat: pd.DataFrame) -> float:
    turnover = pd.to_numeric(strat["turnover"], errors="coerce")
    cost = pd.to_numeric(strat["cost"], errors="coerce")
    valid = (turnover > 0.0) & turnover.notna() & cost.notna()
    if not valid.any():
        return 0.001
    ratios = (cost[valid] / turnover[valid]).astype(float)
    return float(ratios.median())


def build_h3_realism_stress(
    project_root: Path,
    run_backtest: bool = True,
    start_date: str = "1976-10-31",
) -> dict[str, Path]:
    outputs = project_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    if run_backtest:
        _run_baseline_backtest(project_root=project_root, start_date=start_date)

    strat = _load_h3_strategy(project_root=project_root)
    turnover = pd.to_numeric(strat["turnover"], errors="coerce")
    weight = pd.to_numeric(strat["weight"], errors="coerce")
    bh_ret = pd.to_numeric(strat["bh_ret"], errors="coerce")
    cash_ret = pd.to_numeric(strat["cash_ret"], errors="coerce")

    bh_stats = _perf_stats(bh_ret)
    default_cost_rate = _infer_default_cost_rate(strat=strat)

    # Cost sensitivity: keep weights and cash assumption fixed, vary cost-per-turnover.
    cost_scenarios: list[tuple[str, float]] = [
        ("current_default", float(default_cost_rate)),
        ("10bps", 0.0010),
        ("25bps", 0.0025),
        ("50bps", 0.0050),
    ]
    cost_rows: list[dict[str, float | int | str | bool]] = []
    for name, cost_rate in cost_scenarios:
        strat_ret = weight * bh_ret + (1.0 - weight) * cash_ret - float(cost_rate) * turnover
        row = _build_metric_row(
            returns=strat_ret,
            turnover=turnover,
            scenario_type="cost_sensitivity",
            scenario_name=name,
            scenario_value=float(cost_rate),
            bh_stats=bh_stats,
        )
        row["cost_bps_per_turnover"] = float(cost_rate) * 10000.0
        cost_rows.append(row)
    cost_df = pd.DataFrame(cost_rows)

    # Cash sensitivity: keep default cost rate fixed, vary cash yield assumption.
    cash_scenarios: list[tuple[str, float]] = [
        ("current_default_cash", 1.0),
        ("zero_cash", 0.0),
        ("half_cash", 0.5),
    ]
    cash_rows: list[dict[str, float | int | str | bool]] = []
    for name, mult in cash_scenarios:
        effective_cash = cash_ret * float(mult)
        strat_ret = weight * bh_ret + (1.0 - weight) * effective_cash - float(default_cost_rate) * turnover
        row = _build_metric_row(
            returns=strat_ret,
            turnover=turnover,
            scenario_type="cash_sensitivity",
            scenario_name=name,
            scenario_value=float(mult),
            bh_stats=bh_stats,
        )
        row["cash_multiplier"] = float(mult)
        row["cost_bps_per_turnover"] = float(default_cost_rate) * 10000.0
        cash_rows.append(row)
    cash_df = pd.DataFrame(cash_rows)

    keep_cols = [
        "scenario_type",
        "scenario_name",
        "scenario_value",
        "cost_bps_per_turnover",
        "cash_multiplier",
        "months",
        "final_equity",
        "CAGR",
        "vol",
        "Sharpe",
        "max_drawdown",
        "avg_turnover",
        "total_turnover",
        "bh_CAGR",
        "bh_vol",
        "bh_Sharpe",
        "bh_max_drawdown",
        "beats_bh_sharpe",
        "better_dd_than_bh",
    ]
    for col in keep_cols:
        if col not in cost_df.columns:
            cost_df[col] = np.nan
        if col not in cash_df.columns:
            cash_df[col] = np.nan
    cost_df = cost_df[keep_cols]
    cash_df = cash_df[keep_cols]

    cost_out = outputs / "strategy_h3_cost_sensitivity.csv"
    cash_out = outputs / "strategy_h3_cash_sensitivity.csv"
    cost_df.to_csv(cost_out, index=False)
    cash_df.to_csv(cash_out, index=False)

    return {"cost_sensitivity": cost_out, "cash_sensitivity": cash_out}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test h3 baseline realism assumptions.")
    parser.add_argument("--project-root", type=str, default=".", help="Project root path.")
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Use existing outputs/strategy_h3_baseline.csv instead of rerunning backtest.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="1976-10-31",
        help="Backtest start used for max valid no-lookahead history under current pipeline constraints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    artifacts = build_h3_realism_stress(
        project_root=project_root,
        run_backtest=not bool(args.skip_backtest),
        start_date=str(args.start_date),
    )
    print("[h3-realism-stress] wrote:")
    for key, path in artifacts.items():
        print(f" - {key}: {path}")


if __name__ == "__main__":
    main()
