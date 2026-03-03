import pandas as pd

def build_daily_revenue(
    raw_csv_path: str,
    date_col: str,
    qty_col: str,
    price_col: str,
    freq: str = "D",
    fill_missing_days: bool = True,
    drop_returns: bool = True
) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path)

    # Parse datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, qty_col, price_col])

    # Numeric
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[qty_col, price_col])

    if drop_returns:
        df = df[(df[qty_col] > 0) & (df[price_col] > 0)]

    df["revenue"] = df[qty_col] * df[price_col]
    df["date"] = df[date_col].dt.floor(freq)

    kpi = df.groupby("date", as_index=False)["revenue"].sum().sort_values("date")
    kpi = kpi.rename(columns={"revenue": "y"})

    if fill_missing_days:
        full_idx = pd.date_range(kpi["date"].min(), kpi["date"].max(), freq=freq)
        kpi = kpi.set_index("date").reindex(full_idx).fillna(0.0).rename_axis("date").reset_index()

    return kpi