from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame, ma_window: int) -> pd.DataFrame:
    """Build leakage-safe features available at time t to predict t+h."""
    out = df.copy().sort_index()

    # Macro-derived signals built from contemporaneous macro levels.
    if {"GS10", "TB3MS"}.issubset(out.columns):
        out["yc_10y_3m"] = out["GS10"] - out["TB3MS"]
    if {"GS10", "GS2"}.issubset(out.columns):
        out["yc_10y_2y"] = out["GS10"] - out["GS2"]
    if {"BAA", "AAA"}.issubset(out.columns):
        out["credit_spread"] = out["BAA"] - out["AAA"]
    if "CPIAUCSL" in out.columns:
        out["inflation_yoy"] = out["CPIAUCSL"].pct_change(12)
        out["inflation_mom"] = out["CPIAUCSL"].pct_change(1)
    if "INDPRO" in out.columns:
        out["ip_yoy"] = out["INDPRO"].pct_change(12)
    if "UNRATE" in out.columns:
        out["unrate_3m_change"] = out["UNRATE"] - out["UNRATE"].shift(3)

    macro_cols = [c for c in out.columns if c not in {"close", "return_1m"}]

    # Universal 1M information lag: month t uses macro info known at end of t-1.
    if macro_cols:
        print(f"[features] shifting macro columns by 1 month: {macro_cols}")
        out[macro_cols] = out[macro_cols].shift(1)

    # Macro series can be sparse by release timing; fill forward across months only.
    if macro_cols:
        out[macro_cols] = out[macro_cols].ffill()

    out["ret_1m_lag1"] = out["return_1m"].shift(1)
    out["ret_3m_lag1"] = out["return_1m"].rolling(3, min_periods=1).sum().shift(1)
    out[f"ma_{ma_window}_lag1"] = out["close"].rolling(ma_window, min_periods=1).mean().shift(1)
    out["price_to_ma_lag1"] = out["close"].shift(1) / out[f"ma_{ma_window}_lag1"]

    for col in macro_cols:
        out[f"{col}_lag1"] = out[col].shift(1)
        out[f"{col}_lag3"] = out[col].shift(3)

    feature_cols = [
        "ret_1m_lag1",
        "ret_3m_lag1",
        f"ma_{ma_window}_lag1",
        "price_to_ma_lag1",
    ]
    for col in macro_cols:
        feature_cols.append(f"{col}_lag1")
        feature_cols.append(f"{col}_lag3")

    out = out[feature_cols]
    max_lag = 3
    return out.iloc[max_lag:]
