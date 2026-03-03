import numpy as np
import pandas as pd
from xgboost import XGBRegressor

def train_final_model(df_feat: pd.DataFrame, model_params: dict) -> tuple[XGBRegressor, list[str]]:
    feature_cols = [c for c in df_feat.columns if c not in ("date", "y")]
    model = XGBRegressor(**model_params)
    model.fit(df_feat[feature_cols], df_feat["y"])
    return model, feature_cols

def recursive_forecast(
    history: pd.DataFrame,
    model: XGBRegressor,
    lags: list[int],
    rolling_windows: list[int],
    horizon_days: int = 14
) -> pd.DataFrame:
    """
    history: DataFrame with columns ['date','y'] daily, sorted.
    Returns future df with columns ['date','yhat'].
    """
    hist = history.copy().sort_values("date").reset_index(drop=True)
    future_rows = []

    last_date = pd.to_datetime(hist["date"].iloc[-1])

    for h in range(1, horizon_days + 1):
        d = last_date + pd.Timedelta(days=h)

        # build feature row from latest available y (includes previous predictions)
        y_series = hist["y"].astype(float)

        row = {"date": d}
        for lag in lags:
            row[f"lag_{lag}"] = y_series.iloc[-lag] if len(y_series) >= lag else np.nan

        base = y_series.shift(1)
        # rolling computed on base -> but here we can emulate using last values:
        # simplest: compute from y_series excluding current day (which doesn't exist yet)
        y_for_roll = y_series.copy()
        for w in rolling_windows:
            vals = y_for_roll.iloc[-w:] if len(y_for_roll) >= w else y_for_roll
            row[f"roll_mean_{w}"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"roll_std_{w}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        row["dow"] = d.dayofweek
        row["month"] = d.month

        X = pd.DataFrame([row]).drop(columns=["date"])
        yhat = float(model.predict(X)[0])

        future_rows.append({"date": d, "yhat": yhat})

        # append prediction to history to enable recursive lags
        hist = pd.concat([hist, pd.DataFrame([{"date": d, "y": yhat}])], ignore_index=True)

    return pd.DataFrame(future_rows)