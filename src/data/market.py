from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.data.fred import fetch_fred_series, get_fred_api_key


def _safe_ticker_name(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", ticker).strip("_")


def _load_yahoo_daily(
    ticker: str,
    start_date: str,
    end_date: str,
    cache_dir: str = "data/raw/market",
) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for market_source='yahoo'. Install requirements.") from exc

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    market_cache_dir = Path(cache_dir)
    market_cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(market_cache_dir / ".yfinance_tz_cache"))

    cache_path = market_cache_dir / f"{_safe_ticker_name(ticker)}_daily.csv"

    cached = pd.Series(dtype=float)
    if cache_path.exists():
        cached_df = pd.read_csv(cache_path, parse_dates=["date"])
        if not cached_df.empty:
            cached_df = cached_df.sort_values("date").drop_duplicates(subset=["date"])
            cached = cached_df.set_index("date")["close"].astype(float)

    need_download = cached.empty or cached.index.min() > start_ts or cached.index.max() < end_ts
    if need_download:
        # yfinance treats end as exclusive; add one day to include the requested end_date.
        dl_end = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        daily_df = yf.download(
            ticker,
            start=start_ts.strftime("%Y-%m-%d"),
            end=dl_end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            actions=False,
        )
        if daily_df.empty:
            raise RuntimeError(f"No Yahoo data returned for ticker '{ticker}'.")

        if isinstance(daily_df.columns, pd.MultiIndex):
            if ("Close", ticker) in daily_df.columns:
                close_series = daily_df[("Close", ticker)]
            elif "Close" in daily_df.columns.get_level_values(0):
                close_series = daily_df.xs("Close", axis=1, level=0).iloc[:, 0]
            else:
                raise RuntimeError("Yahoo response missing 'Close' column in multi-index frame.")
        else:
            if "Close" not in daily_df.columns:
                raise RuntimeError("Yahoo response missing 'Close' column.")
            close_series = daily_df["Close"]

        downloaded = close_series.rename("close").to_frame().reset_index()
        date_col = downloaded.columns[0]
        downloaded = downloaded.rename(columns={date_col: "date"})
        downloaded["date"] = pd.to_datetime(downloaded["date"], errors="coerce")
        downloaded = downloaded.dropna(subset=["date", "close"])
        downloaded = downloaded.sort_values("date").drop_duplicates(subset=["date"])

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if not cached.empty:
            cached_reset = cached.rename("close").reset_index().rename(columns={"index": "date"})
            merged = pd.concat([cached_reset, downloaded], ignore_index=True)
            merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        else:
            merged = downloaded

        merged.to_csv(cache_path, index=False)
        series = merged.set_index("date")["close"].astype(float)
    else:
        series = cached

    series = series.sort_index()
    return series.loc[(series.index >= start_ts) & (series.index <= end_ts)]


def _load_market_monthly_fred(start_date: str, end_date: str) -> pd.Series:
    # Enforce explicit API configuration for the FRED path.
    get_fred_api_key()
    sp500_daily = fetch_fred_series(
        series_id="SP500",
        start_date=start_date,
        end_date=end_date,
        cache_path="data/raw/fred_SP500_daily.csv",
    )
    if sp500_daily.empty:
        raise RuntimeError("SP500 series returned no observations for requested date range.")
    print(
        "[data] SP500 raw span "
        f"{sp500_daily.index.min().date()} -> {sp500_daily.index.max().date()} "
        f"({len(sp500_daily)} obs)"
    )
    return sp500_daily.resample("ME").last()


def _load_market_monthly_yahoo(ticker: str, start_date: str, end_date: str) -> pd.Series:
    daily_close = _load_yahoo_daily(ticker=ticker, start_date=start_date, end_date=end_date)
    if daily_close.empty:
        raise RuntimeError(f"Yahoo close series is empty for ticker '{ticker}'.")
    print(
        "[data] yahoo raw span "
        f"{daily_close.index.min().date()} -> {daily_close.index.max().date()} "
        f"({len(daily_close)} obs)"
    )
    return daily_close.resample("ME").last()


def build_monthly_dataset(
    start_date: str,
    end_date: str,
    frequency: str = "ME",
    seed: int = 42,
    market_source: str = "yahoo",
    market_ticker: str = "^GSPC",
) -> pd.DataFrame:
    """Create a monthly market dataset from configured source."""
    _ = seed  # Kept for backward-compatible signature with existing CLI.
    freq = "ME" if frequency == "M" else frequency
    if freq != "ME":
        raise ValueError(f"Unsupported frequency '{frequency}'. Use 'ME' for month-end data.")

    source = market_source.lower().strip()
    if source == "yahoo":
        monthly_close_raw = _load_market_monthly_yahoo(
            ticker=market_ticker,
            start_date=start_date,
            end_date=end_date,
        )
    elif source == "fred":
        monthly_close_raw = _load_market_monthly_fred(start_date=start_date, end_date=end_date)
    else:
        raise ValueError(f"Unsupported market_source '{market_source}'. Use 'yahoo' or 'fred'.")

    monthly_index = pd.date_range(start=start_date, end=end_date, freq="ME")
    monthly_close = monthly_close_raw.reindex(monthly_index).ffill()
    print(f"[data] monthly close non-null after reindex/ffill: {int(monthly_close.notna().sum())}")
    df = pd.DataFrame(index=monthly_index)
    df["close"] = monthly_close.astype(float)
    df["return_1m"] = df["close"].pct_change()

    # Keep only the configured date window and ensure month-end index.
    df = df.loc[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
    df.index = df.index.to_period("M").to_timestamp("M")
    df.index.name = "date"
    return df
