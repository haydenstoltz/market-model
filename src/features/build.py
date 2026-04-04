from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_BATCH_1_COLUMNS = [
    "drawdown_12m",
    "months_since_12m_high",
    "realized_vol_3m",
    "vol_ratio_3m_12m",
    "path_efficiency_6m",
    "momentum_12m_ex_1m",
]

FEATURE_BATCH_2_COLUMNS = [
    "tbill_3m_level",
    "tbill_3m_change_3m",
    "treasury_10y_level",
    "term_spread_10y_3m",
    "cpi_yoy",
    "real_short_rate",
    "equity_excess_return_12m",
]


def _expanding_log_trend_residual(price: pd.Series, min_periods: int = 60) -> pd.Series:
    """Expanding OLS residual of log(price) on time index, using data up to t only."""
    if min_periods < 2:
        raise ValueError("min_periods must be at least 2 for trend estimation.")

    log_price = np.log(price.astype(float))
    residual = pd.Series(np.nan, index=log_price.index, dtype=float)
    x_all = np.arange(len(log_price), dtype=float)

    for i in range(len(log_price)):
        y_hist = log_price.iloc[: i + 1]
        valid_mask = y_hist.notna().to_numpy()
        if valid_mask.sum() < min_periods:
            continue

        x_hist = x_all[: i + 1][valid_mask]
        y_vals = y_hist.to_numpy()[valid_mask]

        x_mean = float(x_hist.mean())
        y_mean = float(y_vals.mean())
        sxx = float(np.sum((x_hist - x_mean) ** 2))
        if sxx <= 0.0:
            continue

        beta = float(np.sum((x_hist - x_mean) * (y_vals - y_mean)) / sxx)
        alpha = y_mean - beta * x_mean
        fitted_t = alpha + beta * x_all[i]

        y_t = log_price.iat[i]
        if not np.isnan(y_t):
            residual.iat[i] = float(y_t - fitted_t)

    return residual


def build_features(
    df: pd.DataFrame,
    ma_window: int,
    use_residual_method: bool = False,
    residual_warmup: int = 60,
    use_feature_batch_1: bool = False,
    use_feature_batch_2: bool = False,
) -> pd.DataFrame:
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

    if use_residual_method:
        if "close" not in out.columns:
            raise ValueError("residual_method requires a 'close' price column.")

        # Expanding trend at t uses only history through t, then we lag by 1 month for execution safety.
        residual_raw = _expanding_log_trend_residual(out["close"], min_periods=int(residual_warmup))
        out["residual_method"] = residual_raw.shift(1)

        first_valid = out["residual_method"].first_valid_index()
        print(f"[features] residual_method enabled=True warmup={int(residual_warmup)}")
        if first_valid is None:
            print("[features] residual_method has no valid values after warmup/lag")
        else:
            valid = out.loc[out["residual_method"].notna(), "residual_method"]
            print(
                "[features] residual_method "
                f"first_valid={first_valid.date()} mean={valid.mean():0.6f} std={valid.std(ddof=0):0.6f}"
            )
            print("[features] residual_method preview")
            print(valid.head(3).to_string())
    else:
        print("[features] residual_method enabled=False")

    macro_cols = [c for c in out.columns if c not in {"close", "return_1m", "residual_method"}]

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

    if use_feature_batch_1:
        close = out["close"].astype(float)
        ret_1m = out["return_1m"].astype(float)

        rolling_high_12m = close.rolling(12, min_periods=12).max()
        out["drawdown_12m"] = (close / rolling_high_12m - 1.0).shift(1)

        months_since_high = close.rolling(12, min_periods=12).apply(
            lambda window: float(len(window) - 1 - np.argmax(window)),
            raw=True,
        )
        out["months_since_12m_high"] = months_since_high.shift(1)

        realized_vol_3m_raw = ret_1m.rolling(3, min_periods=3).std(ddof=0)
        realized_vol_12m_raw = ret_1m.rolling(12, min_periods=12).std(ddof=0)
        out["realized_vol_3m"] = realized_vol_3m_raw.shift(1)

        vol_ratio = realized_vol_3m_raw / realized_vol_12m_raw.replace(0.0, np.nan)
        out["vol_ratio_3m_12m"] = vol_ratio.replace([np.inf, -np.inf], np.nan).shift(1)

        trailing_6m_cumret = (1.0 + ret_1m).rolling(6, min_periods=6).apply(np.prod, raw=True) - 1.0
        trailing_6m_abs_sum = ret_1m.abs().rolling(6, min_periods=6).sum()
        out["path_efficiency_6m"] = pd.Series(
            np.where(trailing_6m_abs_sum > 1e-12, trailing_6m_cumret / trailing_6m_abs_sum, 0.0),
            index=out.index,
            dtype=float,
        ).shift(1)

        momentum_11m = (1.0 + ret_1m).rolling(11, min_periods=11).apply(np.prod, raw=True) - 1.0
        # Exclude the most recent month by using one additional lag.
        out["momentum_12m_ex_1m"] = momentum_11m.shift(2)

    created_batch_2_features: list[str] = []
    missing_batch_2_features: list[str] = []
    if use_feature_batch_2:
        ret_1m = out["return_1m"].astype(float)

        tbill_raw = out["TB3MS"].astype(float) if "TB3MS" in out.columns else None
        gs10_raw = out["GS10"].astype(float) if "GS10" in out.columns else None
        cpi_raw = out["CPIAUCSL"].astype(float) if "CPIAUCSL" in out.columns else None

        if tbill_raw is not None:
            out["tbill_3m_level"] = tbill_raw.shift(1)
            out["tbill_3m_change_3m"] = tbill_raw.diff(3).shift(1)
            created_batch_2_features.extend(["tbill_3m_level", "tbill_3m_change_3m"])
        else:
            missing_batch_2_features.extend(["tbill_3m_level", "tbill_3m_change_3m"])

        if gs10_raw is not None:
            out["treasury_10y_level"] = gs10_raw.shift(1)
            created_batch_2_features.append("treasury_10y_level")
        else:
            missing_batch_2_features.append("treasury_10y_level")

        if gs10_raw is not None and tbill_raw is not None:
            out["term_spread_10y_3m"] = (gs10_raw - tbill_raw).shift(1)
            created_batch_2_features.append("term_spread_10y_3m")
        else:
            missing_batch_2_features.append("term_spread_10y_3m")

        if cpi_raw is not None:
            out["cpi_yoy"] = (cpi_raw.pct_change(12) * 100.0).shift(1)
            created_batch_2_features.append("cpi_yoy")
        else:
            missing_batch_2_features.append("cpi_yoy")

        if tbill_raw is not None and cpi_raw is not None:
            cpi_yoy_pct = (cpi_raw.pct_change(12) * 100.0).shift(1)
            out["real_short_rate"] = tbill_raw.shift(1) - cpi_yoy_pct
            created_batch_2_features.append("real_short_rate")
        else:
            missing_batch_2_features.append("real_short_rate")

        if tbill_raw is not None and "return_1m" in out.columns:
            equity_12m = (1.0 + ret_1m).rolling(12, min_periods=12).apply(np.prod, raw=True) - 1.0
            cash_ret_1m = tbill_raw / 100.0 / 12.0
            cash_12m = (1.0 + cash_ret_1m).rolling(12, min_periods=12).apply(np.prod, raw=True) - 1.0
            out["equity_excess_return_12m"] = (equity_12m - cash_12m).shift(1)
            created_batch_2_features.append("equity_excess_return_12m")
        else:
            missing_batch_2_features.append("equity_excess_return_12m")

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

    base_feature_count = len(feature_cols)
    if use_feature_batch_1:
        feature_cols.extend(FEATURE_BATCH_1_COLUMNS)

    if use_feature_batch_2:
        feature_cols.extend(created_batch_2_features)

    if use_residual_method:
        feature_cols.append("residual_method")

    out = out[feature_cols]

    if use_feature_batch_1:
        print(
            "[features] feature_batch_1 enabled=True "
            f"base_feature_count={base_feature_count} final_feature_count={len(feature_cols)}"
        )
        for feature_name in FEATURE_BATCH_1_COLUMNS:
            first_valid = out[feature_name].first_valid_index() if feature_name in out.columns else None
            first_valid_str = "none" if first_valid is None else first_valid.date().isoformat()
            print(
                f"[features] feature_batch_1 {feature_name} "
                f"included={feature_name in out.columns} first_valid={first_valid_str}"
            )
    else:
        print(f"[features] feature_batch_1 enabled=False base_feature_count={base_feature_count}")

    if use_feature_batch_2:
        print(
            "[features] feature_batch_2 enabled=True "
            f"base_feature_count={base_feature_count} final_feature_count={len(feature_cols)}"
        )
        print(f"[features] feature_batch_2 created={created_batch_2_features}")
        print(f"[features] feature_batch_2 missing={missing_batch_2_features}")
        for feature_name in created_batch_2_features:
            first_valid = out[feature_name].first_valid_index() if feature_name in out.columns else None
            first_valid_str = "none" if first_valid is None else first_valid.date().isoformat()
            print(
                f"[features] feature_batch_2 {feature_name} "
                f"included={feature_name in out.columns} first_valid={first_valid_str}"
            )
    else:
        print(f"[features] feature_batch_2 enabled=False base_feature_count={base_feature_count}")

    max_lag = 3
    return out.iloc[max_lag:]
