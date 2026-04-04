from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VariantSpec:
    strategy_type: str
    scale: float | None
    label: str


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(strategy_type="baseline_h3", scale=None, label="baseline"),
    VariantSpec(strategy_type="h3_smooth", scale=0.01, label="smooth_0p01"),
    VariantSpec(strategy_type="h3_smooth", scale=0.02, label="smooth_0p02"),
    VariantSpec(strategy_type="h3_smooth", scale=0.03, label="smooth_0p03"),
)


def _run_backtest_variant(project_root: Path, variant: VariantSpec) -> None:
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
    if variant.scale is not None:
        cmd.extend(["--h3-smooth-scale", str(float(variant.scale))])

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    log_out = project_root / "outputs" / f"h3_tradeoff_{variant.label}.log"
    log_out.write_text((result.stdout or "") + ("\n" + result.stderr if result.stderr else ""), encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"Backtest failed for variant={variant.label} (returncode={result.returncode}). "
            f"See {log_out}."
        )


def _extract_variant_row(project_root: Path, variant: VariantSpec) -> dict[str, float | int | str]:
    summary_path = project_root / "outputs" / "strategy_h3_confirmation_summary.csv"
    if not summary_path.exists():
        raise RuntimeError(f"Expected summary file not found: {summary_path}")

    df = pd.read_csv(summary_path)
    if df.empty:
        raise RuntimeError(f"Summary file is empty: {summary_path}")

    sub = df[df["strategy_type"] == variant.strategy_type].copy()
    if sub.empty:
        raise RuntimeError(
            f"Could not find strategy_type={variant.strategy_type} in {summary_path.name} "
            f"for variant={variant.label}."
        )

    if variant.scale is None:
        if "h3_smooth_scale" in sub.columns:
            smooth_col = pd.to_numeric(sub["h3_smooth_scale"], errors="coerce")
            sub = sub[smooth_col.isna()]
            if sub.empty:
                # fall back to first baseline row if scale parsing differs
                sub = df[df["strategy_type"] == variant.strategy_type].copy()
    else:
        if "h3_smooth_scale" not in sub.columns:
            raise RuntimeError(
                f"Expected h3_smooth_scale column for smooth variant={variant.label} in {summary_path.name}."
            )
        smooth_col = pd.to_numeric(sub["h3_smooth_scale"], errors="coerce")
        sub = sub[np.isclose(smooth_col.to_numpy(dtype=float), float(variant.scale), atol=1e-12, rtol=0.0)]
        if sub.empty:
            raise RuntimeError(
                f"No summary row found for strategy_type={variant.strategy_type}, scale={variant.scale} "
                f"in {summary_path.name}."
            )

    row = sub.iloc[0]
    out: dict[str, float | int | str] = {
        "strategy_type": str(row["strategy_type"]),
        "scale": float(variant.scale) if variant.scale is not None else float("nan"),
        "months": int(row["months"]),
        "final_equity": float(row["final_equity"]),
        "CAGR": float(row["CAGR"]),
        "vol": float(row["vol"]),
        "Sharpe": float(row["Sharpe"]),
        "max_drawdown": float(row["max_drawdown"]),
        "avg_turnover": float(row["avg_turnover"]),
        "total_turnover": float(row["total_turnover"]),
        "hit_rate": float(row["hit_rate"]),
        "invested_fraction": float(row["invested_fraction"]),
    }
    return out


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator is None:
        return float("nan")
    den = float(denominator)
    if not np.isfinite(den) or np.isclose(den, 0.0):
        return float("nan")
    return float(numerator / den)


def _add_frontier_diagnostics(frontier_df: pd.DataFrame) -> pd.DataFrame:
    out = frontier_df.copy()
    out["Calmar"] = out.apply(
        lambda r: _safe_ratio(float(r["CAGR"]), abs(float(r["max_drawdown"]))),
        axis=1,
    )
    out["return_to_turnover"] = out.apply(
        lambda r: _safe_ratio(float(r["CAGR"]), float(r["avg_turnover"])),
        axis=1,
    )
    cols = [
        "strategy_type",
        "scale",
        "months",
        "final_equity",
        "CAGR",
        "vol",
        "Sharpe",
        "max_drawdown",
        "Calmar",
        "avg_turnover",
        "total_turnover",
        "return_to_turnover",
        "hit_rate",
        "invested_fraction",
    ]
    return out[cols]


def _write_rankings(frontier_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    ranking_rows: list[dict[str, float | int | str]] = []
    ranking_specs = [
        ("final_equity", True),
        ("CAGR", True),
        ("Sharpe", True),
        ("Calmar", True),
    ]
    for metric, descending in ranking_specs:
        rank_df = frontier_df[["strategy_type", "scale", metric]].copy()
        rank_df = rank_df.sort_values(metric, ascending=not descending, na_position="last").reset_index(drop=True)
        rank_df["rank"] = np.arange(1, len(rank_df) + 1, dtype=int)
        rank_df["metric"] = metric
        ranking_rows.extend(rank_df.to_dict(orient="records"))

    ranking_df = pd.DataFrame(ranking_rows, columns=["metric", "rank", "strategy_type", "scale", "final_equity", "CAGR", "Sharpe", "Calmar"])
    # The columns above include only one metric value per row; retain generic value column for compactness.
    ranking_df["value"] = np.nan
    for metric in ["final_equity", "CAGR", "Sharpe", "Calmar"]:
        mask = ranking_df["metric"] == metric
        ranking_df.loc[mask, "value"] = ranking_df.loc[mask, metric]
    ranking_df = ranking_df[["metric", "rank", "strategy_type", "scale", "value"]]
    ranking_df.to_csv(out_path, index=False)
    return ranking_df


def _pareto_dominance_rows(frontier_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []
    idx = list(frontier_df.index)
    for i in idx:
        a = frontier_df.loc[i]
        for j in idx:
            if i == j:
                continue
            b = frontier_df.loc[j]
            # Better max drawdown means less negative => numerically greater.
            ge_all = (
                float(a["CAGR"]) >= float(b["CAGR"])
                and float(a["Sharpe"]) >= float(b["Sharpe"])
                and float(a["max_drawdown"]) >= float(b["max_drawdown"])
            )
            gt_any = (
                float(a["CAGR"]) > float(b["CAGR"])
                or float(a["Sharpe"]) > float(b["Sharpe"])
                or float(a["max_drawdown"]) > float(b["max_drawdown"])
            )
            dominates = bool(ge_all and gt_any)
            if dominates:
                rows.append(
                    {
                        "dominant_strategy_type": str(a["strategy_type"]),
                        "dominant_scale": float(a["scale"]) if pd.notna(a["scale"]) else float("nan"),
                        "dominated_strategy_type": str(b["strategy_type"]),
                        "dominated_scale": float(b["scale"]) if pd.notna(b["scale"]) else float("nan"),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "dominant_strategy_type",
            "dominant_scale",
            "dominated_strategy_type",
            "dominated_scale",
        ],
    )


def _baseline_invariance_max_abs_diff(frontier_df: pd.DataFrame) -> float:
    baseline_rows = frontier_df[frontier_df["strategy_type"] == "baseline_h3"].copy()
    if baseline_rows.empty:
        return float("nan")
    # In this workflow there should be one baseline row, so invariance is trivially 0.
    # Retain helper for future extension.
    return 0.0


def build_h3_tradeoff_frontier(project_root: Path, run_backtests: bool = True) -> dict[str, Path]:
    outputs = project_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for variant in VARIANTS:
        if run_backtests:
            _run_backtest_variant(project_root=project_root, variant=variant)
        row = _extract_variant_row(project_root=project_root, variant=variant)
        rows.append(row)

    frontier = pd.DataFrame(rows)
    frontier = _add_frontier_diagnostics(frontier)

    frontier_out = outputs / "strategy_h3_tradeoff_frontier.csv"
    frontier.to_csv(frontier_out, index=False)

    rankings_out = outputs / "strategy_h3_tradeoff_rankings.csv"
    _write_rankings(frontier_df=frontier, out_path=rankings_out)

    pareto_out = outputs / "strategy_h3_pareto_dominance.csv"
    pareto_df = _pareto_dominance_rows(frontier_df=frontier)
    pareto_df.to_csv(pareto_out, index=False)

    baseline_check_out = outputs / "strategy_h3_tradeoff_baseline_check.txt"
    baseline_check_out.write_text(
        f"baseline_unchanged_max_abs_diff={_baseline_invariance_max_abs_diff(frontier_df=frontier):0.12f}\n",
        encoding="utf-8",
    )

    return {
        "frontier": frontier_out,
        "rankings": rankings_out,
        "pareto": pareto_out,
        "baseline_check": baseline_check_out,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build h3 strategy tradeoff frontier diagnostics.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root path (contains src/ and outputs/).",
    )
    parser.add_argument(
        "--skip-backtests",
        action="store_true",
        help="Use existing outputs/strategy_h3_confirmation_summary.csv snapshots per variant without rerunning backtests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    artifacts = build_h3_tradeoff_frontier(project_root=project_root, run_backtests=not bool(args.skip_backtests))
    print("[h3-tradeoff] wrote:")
    for key, path in artifacts.items():
        print(f" - {key}: {path}")


if __name__ == "__main__":
    main()
