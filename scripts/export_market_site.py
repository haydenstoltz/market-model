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

PNG_FILES = [
    "strategy_method_effectiveness.png",
    "strategy_returns_comparison.png",
    "residual_method_over_time.png",
]

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
}


def _coerce_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
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
    return pd.read_csv(path)


def _latest_prediction_cards(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    if predictions is None or predictions.empty:
        return []
    if "horizon" not in predictions.columns or "date" not in predictions.columns:
        return []

    df = predictions.copy()
    df["_date_sort"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["horizon", "_date_sort"])
    if df.empty:
        return []

    cards: list[dict[str, Any]] = []
    for horizon_value, group in df.groupby("horizon", dropna=True):
        latest = group.sort_values("_date_sort").tail(1).drop(columns=["_date_sort"])
        record = _records(latest, limit=1)[0]
        cards.append(record)
    cards.sort(key=lambda x: (x.get("horizon") is None, x.get("horizon")))
    return cards


def _latest_row(df: pd.DataFrame | None) -> dict[str, Any] | None:
    if df is None or df.empty:
        return None
    return _records(df, limit=1)[0]


def _copy_chart_assets() -> list[dict[str, str]]:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    for filename in PNG_FILES:
        src = OUTPUTS_DIR / filename
        if not src.exists():
            continue
        dst = ASSETS_DIR / filename
        shutil.copy2(src, dst)
        copied.append({
            "title": filename.replace("_", " ").replace(".png", "").title(),
            "path": f"assets/{filename}",
        })
    return copied


def build_payload() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    tables: dict[str, list[dict[str, Any]]] = {}
    available_sources: list[str] = []
    warnings: list[str] = []

    loaded: dict[str, pd.DataFrame | None] = {}
    for key, filename in CSV_FILES.items():
        path = OUTPUTS_DIR / filename
        try:
            df = _read_csv_optional(path)
        except Exception as exc:
            df = None
            warnings.append(f"Failed to read outputs/{filename}: {exc}")
        loaded[key] = df
        if df is not None:
            available_sources.append(f"outputs/{filename}")
            if key == "predictions":
                tables["recent_predictions"] = _records(df, limit=60)
            elif key == "strategy_baseline_regime_summary":
                tables[key] = _records(df, limit=36)
            elif key == "ridge_coef_summary":
                tables[key] = _records(df, limit=20)
            else:
                tables[key] = _records(df)

    charts = _copy_chart_assets()
    available_sources.extend([f"docs/{chart['path']}" for chart in charts])

    payload = {
        "status": "ok" if available_sources else "placeholder",
        "message": (
            "Market site payload generated from local outputs/."
            if available_sources
            else "Run the pipeline and then run python scripts/export_market_site.py to populate this page."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "available_sources": available_sources,
        "warnings": warnings,
        "latest_predictions_by_horizon": _latest_prediction_cards(loaded.get("predictions")),
        "latest_positions": {
            "strategy_h1": _latest_row(loaded.get("strategy_h1")),
            "strategy_h1_threshold": _latest_row(loaded.get("strategy_h1_threshold")),
            "strategy_h1_tiered": _latest_row(loaded.get("strategy_h1_tiered")),
        },
        "tables": tables,
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
