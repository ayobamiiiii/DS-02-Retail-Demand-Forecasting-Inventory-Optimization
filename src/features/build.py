import numpy as np
import pandas as pd

LAGS = (1, 7, 14, 28)
def make_features(df):
    """One-step-ahead features: all sales-derived values are shifted first."""
    x = df.sort_values(["id", "date"]).copy()
    g = x.groupby("id", sort=False, observed=True)["sales"]
    for lag in LAGS:
        x[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(1)
    x["roll_mean_7"] = shifted.groupby(x.id, sort=False).rolling(7).mean().reset_index(level=0, drop=True)
    x["roll_mean_28"] = shifted.groupby(x.id, sort=False).rolling(28).mean().reset_index(level=0, drop=True)
    x["roll_std_28"] = shifted.groupby(x.id, sort=False).rolling(28).std().reset_index(level=0, drop=True)
    x["dayofweek"] = x.date.dt.dayofweek
    x["month"] = x.date.dt.month
    x["year"] = x.date.dt.year
    x["day"] = x.date.dt.day
    x["is_weekend"] = (x.dayofweek >= 5).astype(int)
    for c in ("event_name_1", "event_type_1", "event_name_2", "event_type_2"):
        if c in x: x[c] = x[c].fillna("none").astype("category").cat.codes
    x["snap"] = x[[c for c in ("snap_CA", "snap_TX", "snap_WI") if c in x]].max(axis=1) if any(c in x for c in ("snap_CA", "snap_TX", "snap_WI")) else 0
    x["price_change"] = x.groupby("id", observed=True).sell_price.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0) if "sell_price" in x else 0
    return x

FEATURES = ["lag_1", "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28", "roll_std_28",
            "dayofweek", "month", "year", "day", "is_weekend", "snap", "price_change", "sell_price"]
