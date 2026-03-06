from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


def load_synthetic_fred(index: pd.DatetimeIndex, seed: int = 42) -> pd.DataFrame:
    """Return synthetic macro series at monthly frequency."""
    rng = np.random.default_rng(seed)
    n = len(index)

    inflation = 0.002 + 0.0008 * np.sin(np.linspace(0, 8, n)) + rng.normal(0, 0.0006, n)
    unemployment = 0.05 + 0.01 * np.sin(np.linspace(0, 5, n) + 1.0) + rng.normal(0, 0.002, n)
    term_spread = 0.01 + 0.005 * np.cos(np.linspace(0, 7, n)) + rng.normal(0, 0.0015, n)

    return pd.DataFrame(
        {
            "inflation_mom": inflation,
            "unemployment_rate": unemployment.clip(0.02, 0.15),
            "term_spread": term_spread,
        },
        index=index,
    )


def get_fred_api_key() -> str:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Export FRED_API_KEY before running data ingestion."
        )
    return api_key


def fetch_fred_series(
    series_id: str,
    start_date: str,
    end_date: str,
    cache_path: str | Path | None = None,
) -> pd.Series:
    """Fetch a FRED series with optional local CSV caching."""
    api_key = get_fred_api_key()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            cached = pd.read_csv(cache_path, parse_dates=["date"])
            cached = cached.set_index("date").sort_index()
            if not cached.empty:
                cache_min = cached.index.min()
                cache_max = cached.index.max()
                if cache_min <= start_ts and cache_max >= end_ts:
                    series = cached["value"].rename(series_id)
                    return series.loc[(series.index >= start_ts) & (series.index <= end_ts)]

    base_url = "https://api.stlouisfed.org/fred/series/observations"
    limit = 100000
    offset = 0
    all_observations: list[dict] = []
    total_count: int | None = None

    while True:
        response = requests.get(
            base_url,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": start_date,
                "observation_end": end_date,
                "limit": limit,
                "offset": offset,
                "sort_order": "asc",
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        if total_count is None and payload.get("count") is not None:
            total_count = int(payload["count"])

        page_obs = payload.get("observations", [])
        if not page_obs:
            break

        all_observations.extend(page_obs)
        offset += len(page_obs)

        if total_count is not None:
            if len(all_observations) >= total_count:
                break
        elif len(page_obs) < limit:
            break

    if not all_observations:
        raise RuntimeError(f"No observations returned for FRED series '{series_id}'.")

    frame = pd.DataFrame(all_observations)[["date", "value"]]
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"]).sort_values("date").drop_duplicates(subset=["date"])
    if frame.empty:
        raise RuntimeError(f"All observations for FRED series '{series_id}' are missing/non-numeric.")

    print(
        "[fred] fetched "
        f"{len(frame)} obs for {series_id}: "
        f"{frame['date'].min().date()} -> {frame['date'].max().date()}"
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index=False)

    series = frame.set_index("date")["value"].rename(series_id)
    return series.loc[(series.index >= start_ts) & (series.index <= end_ts)]


def load_macro_monthly(
    start_date: str,
    end_date: str,
    series_ids: Iterable[str],
    frequency: str = "ME",
) -> pd.DataFrame:
    """Fetch configured FRED macro series and return month-end dataframe."""
    freq = "ME" if frequency == "M" else frequency
    if freq != "ME":
        raise ValueError(f"Unsupported frequency '{frequency}'. Use 'ME' for month-end data.")
    series_ids = list(series_ids)
    if not series_ids:
        raise ValueError("macro_series_ids config is empty; provide at least one FRED macro series ID.")

    get_fred_api_key()
    out = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq="ME"))

    for series_id in series_ids:
        daily = fetch_fred_series(
            series_id=series_id,
            start_date=start_date,
            end_date=end_date,
            cache_path=f"data/raw/fred_{series_id}.csv",
        )
        monthly = daily.resample("ME").last().dropna()
        out = out.join(monthly.rename(series_id), how="outer")

    out = out.sort_index()
    out = out.ffill()
    out = out.loc[(out.index >= start_date) & (out.index <= end_date)]
    out.index = out.index.to_period("M").to_timestamp("M")
    out.index.name = "date"
    return out
