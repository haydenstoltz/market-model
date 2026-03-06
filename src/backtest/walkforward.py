from __future__ import annotations

import time

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.baselines import make_model


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame


def _with_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
    else:
        out = out.reset_index()
        first_col = out.columns[0]
        out = out.rename(columns={first_col: "date"})
        out["date"] = pd.to_datetime(out["date"])
    return out


def run_walkforward_backtest(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    horizons: list[int],
    train_min_periods: int,
    model_name: str = "ridge",
    start: str | None = None,
    end: str | None = None,
) -> BacktestResult:
    """Expanding-window walk-forward with strict train<test date separation."""
    feat_df = _with_date_column(features)
    targ_df = _with_date_column(targets)
    joined = feat_df.merge(targ_df, on="date", how="inner").sort_values("date").set_index("date")
    if joined.empty:
        raise RuntimeError("Merged modeling frame is empty after joining features and targets on date.")

    if start:
        joined = joined[joined.index >= pd.to_datetime(start)]
    if end:
        joined = joined[joined.index <= pd.to_datetime(end)]
    if joined.empty:
        raise RuntimeError("Modeling frame is empty after applying start/end date filters.")

    print(
        "[backtest] modeling frame "
        f"{joined.index.min().date()} -> {joined.index.max().date()} "
        f"({len(joined)} rows)"
    )

    pred_rows: list[dict[str, float | int | str]] = []
    feature_cols = [c for c in joined.columns if not c.startswith("target_h")]
    feature_cols = [c for c in feature_cols if joined[c].notna().any()]
    if not feature_cols:
        raise RuntimeError("No usable feature columns found after filtering all-NaN features.")

    start_total = time.time()

    for h in horizons:
        target_col = f"target_h{h}"
        if target_col not in joined.columns:
            raise ValueError(f"Missing target column '{target_col}' in modeling frame.")
        printed_debug = False

        max_i = len(joined) - h
        if max_i <= train_min_periods:
            continue

        for window_idx, i in enumerate(range(train_min_periods, max_i)):
            step_start = time.time()
            test_date = joined.index[i]
            y_true_val = joined.iloc[i][target_col]
            if pd.isna(y_true_val):
                step_end = time.time()
                print(f"[backtest] horizon {h} window {window_idx} took {step_end - step_start:.2f}s")
                continue

            train_slice = joined.iloc[:i]
            X_test = joined.iloc[[i]][feature_cols]

            train_y = train_slice[target_col]
            mask = ~train_y.isna()
            if mask.sum() == 0:
                step_end = time.time()
                print(f"[backtest] horizon {h} window {window_idx} took {step_end - step_start:.2f}s")
                continue

            X_train = train_slice.loc[mask, feature_cols]
            y_train = train_y.loc[mask].to_numpy(dtype=float)

            # Drop columns that are all NaN in the current training window.
            valid_cols = X_train.columns[X_train.notna().any()]
            if len(valid_cols) == 0:
                step_end = time.time()
                print(f"[backtest] horizon {h} window {window_idx} took {step_end - step_start:.2f}s")
                continue
            X_train = X_train[valid_cols]
            X_test = X_test[valid_cols]

            if not printed_debug:
                x_train_hash = hashlib.md5(
                    pd.util.hash_pandas_object(X_train, index=True).values.tobytes()
                ).hexdigest()
                x_test_hash = hashlib.md5(
                    pd.util.hash_pandas_object(X_test, index=True).values.tobytes()
                ).hexdigest()
                print(f"[backtest][debug] horizon={h}")
                print(f"[backtest][debug] model={model_name}")
                print(f"[backtest][debug] feature_count={X_train.shape[1]}")
                print(f"[backtest][debug] feature_head={list(X_train.columns)[:20]}")
                print(f"[backtest][debug] X_train_hash={x_train_hash}")
                print(f"[backtest][debug] X_test_hash={x_test_hash}")
                printed_debug = True

            model = make_model(model_name)
            model.fit(X_train, y_train)
            y_pred = float(model.predict(X_test)[0])
            y_true = float(y_true_val)

            pred_rows.append(
                {
                    "date": test_date.strftime("%Y-%m-%d"),
                    "horizon": h,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "error": y_true - y_pred,
                }
            )
            step_end = time.time()
            print(f"[backtest] horizon {h} window {window_idx} took {step_end - step_start:.2f}s")

    print(f"[backtest] total runtime: {time.time() - start_total:.2f}s")

    pred_df = pd.DataFrame(pred_rows)
    if pred_df.empty:
        raise RuntimeError("No predictions produced by walk-forward backtest.")
    pred_df = pred_df.sort_values(["date", "horizon"]).reset_index(drop=True)

    metrics_rows = []
    for h in horizons:
        sub = pred_df[pred_df["horizon"] == h]
        mae = float(np.mean(np.abs(sub["error"])))
        rmse = float(math.sqrt(np.mean(np.square(sub["error"]))))
        metrics_rows.append({"horizon": h, "mae": mae, "rmse": rmse, "n": int(len(sub))})

    metrics_df = pd.DataFrame(metrics_rows)
    return BacktestResult(predictions=pred_df, metrics=metrics_df)
