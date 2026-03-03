import pandas as pd

def add_features(
    kpi: pd.DataFrame,
    lags: list[int],
    rolling_windows: list[int],
    add_calendar: bool = True
) -> pd.DataFrame:
    df = kpi.copy()
    df = df.sort_values("date")

    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    base = df["y"].shift(1)
    for w in rolling_windows:
        df[f"roll_mean_{w}"] = base.rolling(w).mean()
        df[f"roll_std_{w}"] = base.rolling(w).std()

    if add_calendar:
        df["dow"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month

    df = df.dropna().reset_index(drop=True)
    return df