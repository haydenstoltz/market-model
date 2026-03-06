from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(model_name: str = "ridge"):
    name = model_name.lower()

    if name == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])

    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )

    raise ValueError(f"Unknown model: {model_name}")
