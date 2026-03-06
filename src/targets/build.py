from __future__ import annotations

import numpy as np
import pandas as pd


def build_targets(df: pd.DataFrame, horizons: list[int], ma_window: int) -> pd.DataFrame:
    """Build residual-change targets from log-close for each horizon in months."""
    if "close" not in df.columns:
        raise ValueError("Input dataframe must include a 'close' column.")
    if ma_window < 1:
        raise ValueError("ma_window must be >= 1.")

    out = pd.DataFrame(index=df.index)
    close = pd.to_numeric(df["close"], errors="coerce")
    log_close = np.log(close)
    trend = log_close.rolling(ma_window, min_periods=ma_window).mean()
    residual = log_close - trend

    for h in horizons:
        out[f"target_h{h}"] = residual.shift(-h) - residual

    return out
