from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"
DATA_DIR = DOCS_DIR / "data"
ASSETS_DIR = DOCS_DIR / "assets"

CSV_FILES = {
    "predictions": "predictions.csv",
    "strategy_h1": "strategy_h1.csv",
    "strategy_h1_threshold": "strategy_h1_threshold.csv",
    "strategy_h1_tiered": "strategy_h1_tiered.csv",
    "strategy_run_summary": "strategy_run_summary.csv",
    "strategy_baseline_horizon_summary": "strategy_baseline_horizon_summary.csv",
    "strategy_baseline_regime_summary": "strategy_baseline_regime_summary.csv",
    "strategy_h3_confirmation_summary": "strategy_h3_confirmation_summary.csv",
    "ridge_coef_summary": "ridge_coef_summary.csv",
    "strategy_h3_subperiod_summary_fullhistory": "strategy_h3_subperiod_summary_fullhistory.csv",
}

STRATEGY_CURVE_FILES = {
    "baseline_binary": "strategy_h1.csv",
    "thresholded": "strategy_h1_threshold.csv",
    "tiered": "strategy_h1_tiered.csv",
}

STRATEGY_LABELS = {
    "baseline_binary": "Baseline Binary",
    "thresholded": "Thresholded",
    "tiered": "Tiered",
    "buy_and_hold": "Buy & Hold",
}


def _coerce_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    out = df.copy()
    if limit is not None:
        out = out.tail(int(limit))
    out = out.where(pd.notna(out), None)
    records: list[dict[str, Any]] = []
    for row in out.to_dict(orient="records"):
        records.append({str(k): _coerce_scalar(v) for k, v in row.items()})
    return records


def _read_csv_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _safe_sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        return df.copy()
    out = df.copy()
    out = out.dropna(subset=["date"])
    return out.sort_values("date")


def _latest_row(df: pd.DataFrame | None) -> dict[str, Any] | None:
    if df is None or df.empty:
        return None
    out = _safe_sort_by_date(df)
    if out.empty:
        out = df
    return _records(out, limit=1)[0]


def _latest_prediction_cards(predictions: pd.DataFrame | None) -> list[dict[str, Any]]:
    if predictions is None or predictions.empty:
        return []
    if "horizon" not in predictions.columns or "date" not in predictions.columns:
        return []

    df = _safe_sort_by_date(predictions)
    df = df.dropna(subset=["horizon"])
    if df.empty:
        return []

    cards: list[dict[str, Any]] = []
    for horizon_value, group in df.groupby("horizon", dropna=True):
        latest = group.tail(1).copy()
        latest["signal"] = (latest["y_pred"] > 0).astype(int) if "y_pred" in latest.columns else None
        record = _records(latest, limit=1)[0]
        cards.append(record)

    cards.sort(key=lambda x: x.get("horizon"))
    return cards


def _build_predictions_by_horizon(predictions: pd.DataFrame | None) -> dict[str, list[dict[str, Any]]]:
    if predictions is None or predictions.empty:
        return {}
    if "horizon" not in predictions.columns or "date" not in predictions.columns:
        return {}

    df = _safe_sort_by_date(predictions).dropna(subset=["horizon"])
    if df.empty:
        return {}

    payload: dict[str, list[dict[str, Any]]] = {}
    for horizon_value, group in df.groupby("horizon", dropna=True):
        g = group.copy()
        if "y_pred" in g.columns:
            g["signal_pred"] = (g["y_pred"] > 0).astype(int)
        if "y_true" in g.columns:
            g["signal_true"] = (g["y_true"] > 0).astype(int)
        key = f"h{int(horizon_value)}" if pd.notna(horizon_value) else "h?"
        keep = [c for c in ["date", "y_pred", "y_true", "error", "signal_pred", "signal_true"] if c in g.columns]
        payload[key] = _records(g[keep], limit=None)
    return payload


def _build_rolling_hit_rate_by_horizon(predictions: pd.DataFrame | None, window_months: int = 24) -> dict[str, Any]:
    if predictions is None or predictions.empty:
        return {"window_months": int(window_months), "horizons": {}}
    if "horizon" not in predictions.columns or "date" not in predictions.columns:
        return {"window_months": int(window_months), "horizons": {}}
    if "y_pred" not in predictions.columns or "y_true" not in predictions.columns:
        return {"window_months": int(window_months), "horizons": {}}

    df = _safe_sort_by_date(predictions).dropna(subset=["horizon", "y_pred", "y_true"])
    if df.empty:
        return {"window_months": int(window_months), "horizons": {}}

    min_periods = max(6, int(window_months // 3))
    out: dict[str, list[dict[str, Any]]] = {}
    for horizon_value, group in df.groupby("horizon", dropna=True):
        g = group.copy()
        g["hit"] = ((g["y_pred"] > 0) == (g["y_true"] > 0)).astype(float)
        g["rolling_hit_rate"] = g["hit"].rolling(window=int(window_months), min_periods=min_periods).mean()
        g = g.dropna(subset=["rolling_hit_rate"])
        key = f"h{int(horizon_value)}"
        out[key] = _records(g[["date", "rolling_hit_rate"]], limit=None)

    return {"window_months": int(window_months), "horizons": out}


def _series_drawdown(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    running_max = None
    for row in points:
        value = row.get("value")
        if value is None:
            continue
        try:
            v = float(value)
        except Exception:
            continue
        if running_max is None:
            running_max = v
        else:
            running_max = max(running_max, v)
        dd = (v / running_max) - 1.0 if running_max else 0.0
        out.append({"date": row.get("date"), "value": dd})
    return out


def _build_strategy_series(loaded: dict[str, pd.DataFrame | None]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    curve_series: list[dict[str, Any]] = []
    drawdown_series: list[dict[str, Any]] = []
    weight_series: list[dict[str, Any]] = []

    buy_hold_points: list[dict[str, Any]] | None = None
    buy_hold_drawdowns: list[dict[str, Any]] | None = None

    for strategy_name, csv_name in STRATEGY_CURVE_FILES.items():
        df = loaded.get(Path(csv_name).stem)
        if df is None or df.empty or "date" not in df.columns:
            continue

        g = _safe_sort_by_date(df)
        if g.empty:
            continue

        if "equity_curve_strat" in g.columns:
            points = _records(g[["date", "equity_curve_strat"]].rename(columns={"equity_curve_strat": "value"}))
            curve_series.append({
                "strategy": strategy_name,
                "label": STRATEGY_LABELS.get(strategy_name, strategy_name),
                "data": points,
            })
            drawdown_series.append({
                "strategy": strategy_name,
                "label": STRATEGY_LABELS.get(strategy_name, strategy_name),
                "data": _series_drawdown(points),
            })

        if buy_hold_points is None and "equity_curve_bh" in g.columns:
            buy_hold_points = _records(g[["date", "equity_curve_bh"]].rename(columns={"equity_curve_bh": "value"}))
            buy_hold_drawdowns = _series_drawdown(buy_hold_points)

        if "weight" in g.columns:
            weight_points = _records(g[["date", "weight"]].rename(columns={"weight": "value"}))
            weight_series.append({
                "strategy": strategy_name,
                "label": STRATEGY_LABELS.get(strategy_name, strategy_name),
                "data": weight_points,
            })

    if buy_hold_points is not None:
        curve_series.append({
            "strategy": "buy_and_hold",
            "label": STRATEGY_LABELS["buy_and_hold"],
            "is_benchmark": True,
            "data": buy_hold_points,
        })
    if buy_hold_drawdowns is not None:
        drawdown_series.append({
            "strategy": "buy_and_hold",
            "label": STRATEGY_LABELS["buy_and_hold"],
            "is_benchmark": True,
            "data": buy_hold_drawdowns,
        })

    return (
        {"series": curve_series},
        {"series": drawdown_series},
        {"series": weight_series},
    )


def _build_top_coefficients(ridge_coef_summary: pd.DataFrame | None, top_n: int = 20) -> list[dict[str, Any]]:
    if ridge_coef_summary is None or ridge_coef_summary.empty:
        return []
    if "mean_abs_coef" not in ridge_coef_summary.columns:
        return _records(ridge_coef_summary.head(top_n))

    g = ridge_coef_summary.copy()
    g = g.sort_values("mean_abs_coef", ascending=False).head(int(top_n))
    keep = [c for c in ["feature_name", "mean_abs_coef", "mean_coef", "std_coef", "fraction_nontrivial_coef"] if c in g.columns]
    return _records(g[keep], limit=None)


def _best_strategy_snapshot(strategy_run_summary: pd.DataFrame | None, horizon_summary: pd.DataFrame | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "best_risk_adjusted": None,
        "best_growth": None,
    }

    if strategy_run_summary is not None and not strategy_run_summary.empty:
        s = strategy_run_summary.copy()
        if "Sharpe_strat" in s.columns:
            best_sharpe = s.sort_values("Sharpe_strat", ascending=False).head(1)
            out["best_risk_adjusted"] = _records(best_sharpe, limit=1)[0]
        if "final_equity_strat" in s.columns:
            best_growth = s.sort_values("final_equity_strat", ascending=False).head(1)
            out["best_growth"] = _records(best_growth, limit=1)[0]

    if out["best_risk_adjusted"] is None and horizon_summary is not None and not horizon_summary.empty and "Sharpe" in horizon_summary.columns:
        out["best_risk_adjusted"] = _records(horizon_summary.sort_values("Sharpe", ascending=False).head(1), limit=1)[0]
    if out["best_growth"] is None and horizon_summary is not None and not horizon_summary.empty and "final_equity" in horizon_summary.columns:
        out["best_growth"] = _records(horizon_summary.sort_values("final_equity", ascending=False).head(1), limit=1)[0]

    return out


def _live_widget_config() -> dict[str, Any]:
    return {
        "provider": "tradingview",
        "ticker_tape": {
            "script_url": "https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js",
            "config": {
                "symbols": [
                    {"proName": "FOREXCOM:SPXUSD", "title": "S&P 500"},
                    {"proName": "NASDAQ:QQQ", "title": "Nasdaq 100"},
                    {"proName": "AMEX:TLT", "title": "20Y Treasury"},
                    {"proName": "TVC:DXY", "title": "US Dollar"},
                    {"proName": "TVC:GOLD", "title": "Gold"},
                    {"proName": "AMEX:USO", "title": "Crude Oil"},
                ],
                "showSymbolLogo": True,
                "isTransparent": True,
                "displayMode": "adaptive",
                "colorTheme": "dark",
                "locale": "en",
            },
        },
        "main_chart": {
            "script_url": "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js",
            "config": {
                "autosize": True,
                "symbol": "AMEX:SPY",
                "interval": "D",
                "timezone": "America/New_York",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "withdateranges": True,
                "allow_symbol_change": True,
                "save_image": False,
                "hide_top_toolbar": False,
                "hide_legend": False,
                "support_host": "https://www.tradingview.com",
            },
        },
        "market_overview": {
            "script_url": "https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js",
            "config": {
                "colorTheme": "dark",
                "dateRange": "12M",
                "showChart": True,
                "locale": "en",
                "largeChartUrl": "",
                "isTransparent": True,
                "showSymbolLogo": True,
                "showFloatingTooltip": False,
                "plotLineColorGrowing": "rgba(34, 197, 94, 1)",
                "plotLineColorFalling": "rgba(248, 113, 113, 1)",
                "gridLineColor": "rgba(148, 163, 184, 0.2)",
                "scaleFontColor": "rgba(226, 232, 240, 0.85)",
                "belowLineFillColorGrowing": "rgba(34, 197, 94, 0.12)",
                "belowLineFillColorFalling": "rgba(248, 113, 113, 0.12)",
                "belowLineFillColorGrowingBottom": "rgba(34, 197, 94, 0.02)",
                "belowLineFillColorFallingBottom": "rgba(248, 113, 113, 0.02)",
                "tabs": [
                    {
                        "title": "US",
                        "symbols": [
                            {"s": "FOREXCOM:SPXUSD", "d": "S&P 500"},
                            {"s": "NASDAQ:QQQ", "d": "Nasdaq 100"},
                            {"s": "AMEX:DIA", "d": "Dow 30"},
                            {"s": "AMEX:IWM", "d": "Russell 2000"},
                            {"s": "CBOE:VIX", "d": "VIX"},
                        ],
                    },
                    {
                        "title": "Rates & Credit",
                        "symbols": [
                            {"s": "TVC:US10Y", "d": "US 10Y"},
                            {"s": "TVC:US02Y", "d": "US 2Y"},
                            {"s": "AMEX:TLT", "d": "TLT"},
                            {"s": "AMEX:LQD", "d": "IG Credit"},
                            {"s": "AMEX:HYG", "d": "HY Credit"},
                        ],
                    },
                ],
            },
        },
    }


def _copy_chart_assets() -> list[dict[str, str]]:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    for src in sorted(OUTPUTS_DIR.glob("*.png")):
        dst = ASSETS_DIR / src.name
        try:
            shutil.copy2(src, dst)
        except Exception:
            continue
        copied.append(
            {
                "title": src.stem.replace("_", " ").title(),
                "path": f"assets/{src.name}",
            }
        )
    return copied


def _table_payload(loaded: dict[str, pd.DataFrame | None]) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}

    pred_df = loaded.get("predictions")
    if pred_df is not None and not pred_df.empty:
        tables["recent_predictions"] = _records(_safe_sort_by_date(pred_df), limit=120)

    default_limits = {
        "strategy_run_summary": 20,
        "strategy_baseline_horizon_summary": 20,
        "strategy_h3_confirmation_summary": 20,
        "strategy_baseline_regime_summary": 120,
        "ridge_coef_summary": 40,
        "strategy_h3_subperiod_summary_fullhistory": 20,
    }
    for table_key, limit in default_limits.items():
        df = loaded.get(table_key)
        if df is None or df.empty:
            continue
        if "date" in df.columns:
            tables[table_key] = _records(_safe_sort_by_date(df), limit=limit)
        else:
            tables[table_key] = _records(df, limit=limit)

    return tables


def _latest_positions_payload(loaded: dict[str, pd.DataFrame | None]) -> dict[str, Any]:
    positions: dict[str, Any] = {}
    for key in ["strategy_h1", "strategy_h1_threshold", "strategy_h1_tiered"]:
        latest = _latest_row(loaded.get(key))
        if latest is None:
            continue
        if "weight" in latest:
            latest["target_state"] = "INVEST" if float(latest["weight"]) > 0 else "CASH"
        positions[key] = latest
    return positions


def build_payload() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    available_sources: list[str] = []
    loaded: dict[str, pd.DataFrame | None] = {}

    for key, filename in CSV_FILES.items():
        path = OUTPUTS_DIR / filename
        df = _read_csv_optional(path)
        loaded[key] = df
        if df is None:
            warnings.append(f"Missing or unreadable outputs/{filename}")
        else:
            available_sources.append(f"outputs/{filename}")

    charts = _copy_chart_assets()
    available_sources.extend([f"docs/{chart['path']}" for chart in charts])

    predictions = loaded.get("predictions")
    latest_predictions = _latest_prediction_cards(predictions)
    latest_prediction_state = None
    for preferred_horizon in [3, 1, 6]:
        match = next((r for r in latest_predictions if int(r.get("horizon", -1)) == preferred_horizon), None)
        if match is not None:
            latest_prediction_state = match
            break
    if latest_prediction_state is None and latest_predictions:
        latest_prediction_state = latest_predictions[0]

    positions = _latest_positions_payload(loaded)
    best_strategy = _best_strategy_snapshot(
        strategy_run_summary=loaded.get("strategy_run_summary"),
        horizon_summary=loaded.get("strategy_baseline_horizon_summary"),
    )

    strategy_curves, strategy_drawdowns, strategy_weights = _build_strategy_series(loaded)
    top_coefficients = _build_top_coefficients(loaded.get("ridge_coef_summary"), top_n=20)

    payload = {
        "status": "ok" if available_sources else "placeholder",
        "message": (
            "Institutional dashboard payload generated from local outputs/."
            if available_sources
            else "Run pipeline exports and then python scripts/export_market_site.py to populate this dashboard."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "available_sources": available_sources,
        "warnings": warnings,
        "snapshot_metadata": {
            "model_version": "ridge_baseline_export",
            "payload_version": "2.0",
            "source_count": len(available_sources),
            "latest_prediction_state": latest_prediction_state,
            "best_strategy": best_strategy,
        },
        "latest_predictions_by_horizon": latest_predictions,
        "latest_positions": positions,
        "tables": _table_payload(loaded),
        "chart_series": {
            "predictions_by_horizon": _build_predictions_by_horizon(predictions),
            "rolling_hit_rate_by_horizon": _build_rolling_hit_rate_by_horizon(predictions, window_months=24),
            "strategy_curves": strategy_curves,
            "strategy_drawdowns": strategy_drawdowns,
            "strategy_weights": strategy_weights,
            "top_coefficients": top_coefficients,
        },
        "live_widgets": _live_widget_config(),
        "charts": charts,
    }
    return payload


def main() -> None:
    payload = build_payload()
    out_path = DATA_DIR / "market.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[site] wrote {out_path}")


if __name__ == "__main__":
    main()
