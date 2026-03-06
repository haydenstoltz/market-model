from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest.walkforward import run_walkforward_backtest
from src.data.fred import load_macro_monthly
from src.data.market import build_monthly_dataset
from src.features.build import build_features
from src.targets.build import build_targets


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


def run_features(cfg: dict) -> None:
    raw_path = Path(cfg["paths"]["raw_data"])
    df = pd.read_csv(raw_path, parse_dates=["date"], index_col="date")

    feats = build_features(df=df, ma_window=int(cfg["ma_window"]))
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
) -> None:
    feature_path = Path(cfg["paths"]["features"])
    target_path = Path(cfg["paths"]["targets"])
    raw_path = Path(cfg["paths"]["raw_data"])

    features = pd.read_csv(feature_path, parse_dates=["date"], index_col="date")
    targets = pd.read_csv(target_path, parse_dates=["date"], index_col="date")
    raw = pd.read_csv(raw_path, parse_dates=["date"], index_col="date")

    result = run_walkforward_backtest(
        features=features,
        targets=targets,
        horizons=list(cfg["horizons"]),
        train_min_periods=int(cfg.get("train_min_periods", 36)),
        model_name=model_name,
        start=start,
        end=end,
    )

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

    strat = pd.DataFrame(index=aligned.index)
    strat["signal"] = (aligned["y_pred"] > 0.0).astype(int)
    strat["weight"] = strat["signal"].shift(1).fillna(0.0)
    strat["cash_ret"] = aligned["cash_ret"]
    prev_weight = strat["weight"].shift(1).fillna(0.0)
    strat["turnover"] = (strat["weight"] - prev_weight).abs()
    strat["cost"] = 0.0010 * strat["turnover"]  # 10 bps cost when allocation flips
    strat["strat_ret_gross"] = strat["weight"] * aligned["return_1m"] + (1.0 - strat["weight"]) * strat["cash_ret"]
    strat["strat_ret"] = strat["strat_ret_gross"] - strat["cost"]
    strat["bh_ret"] = aligned["return_1m"]
    strat["equity_curve_strat"] = (1.0 + strat["strat_ret"]).cumprod()
    strat["equity_curve_bh"] = (1.0 + strat["bh_ret"]).cumprod()
    strat = strat.reset_index().rename(columns={"index": "date"})

    strat_out = ensure_parent("outputs/strategy_h1.csv")
    strat.to_csv(strat_out, index=False)

    strat_stats = _perf_stats(strat["strat_ret"])
    bh_stats = _perf_stats(strat["bh_ret"])

    print("[backtest] strategy_h1")
    print(f" sign_hit_rate_nonzero: {sign_hit_rate:0.6f}")
    print(
        " strategy: "
        f"CAGR={strat_stats['cagr']:0.6f}, Vol={strat_stats['vol']:0.6f}, "
        f"Sharpe={strat_stats['sharpe']:0.6f}, MaxDD={strat_stats['max_dd']:0.6f}, "
        f"AvgTurnover={strat['turnover'].mean():0.6f}, AvgCost={strat['cost'].mean():0.6f}"
    )
    print(
        " buy_hold: "
        f"CAGR={bh_stats['cagr']:0.6f}, Vol={bh_stats['vol']:0.6f}, "
        f"Sharpe={bh_stats['sharpe']:0.6f}, MaxDD={bh_stats['max_dd']:0.6f}"
    )
    print(f"[backtest] wrote strategy -> {strat_out}")


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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.command == "data":
        run_data(cfg)
    elif args.command == "features":
        run_features(cfg)
    elif args.command == "targets":
        run_targets(cfg)
    elif args.command == "backtest":
        run_backtest(
            cfg,
            model_name=args.model,
            start=args.start,
            end=args.end,
        )


if __name__ == "__main__":
    main()
