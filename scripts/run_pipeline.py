import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocessing.build_kpi import build_daily_revenue
from src.features.make_features import add_features
from src.modeling.evaluate import evaluate_time_cv, save_metrics
from src.modeling.evaluate import evaluate_time_cv, save_metrics, save_last_fold_predictions
from src.reporting.plot_last_fold import plot_last_fold
from src.modeling.predict import train_final_model, recursive_forecast

def main():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    kpi = build_daily_revenue(
        raw_csv_path=cfg["dataset"]["raw_csv_path"],
        date_col=cfg["dataset"]["date_col"],
        qty_col=cfg["dataset"]["qty_col"],
        price_col=cfg["dataset"]["price_col"],
        freq=cfg["kpi"]["freq"],
        fill_missing_days=cfg["kpi"]["fill_missing_days"],
        drop_returns=cfg["kpi"]["drop_returns"],
    )
    kpi.to_csv("data/processed/daily_kpi.csv", index=False)

    df_feat = add_features(
        kpi=kpi,
        lags=cfg["features"]["lags"],
        rolling_windows=cfg["features"]["rolling_windows"],
        add_calendar=cfg["features"]["add_calendar"],
    )
    df_feat.to_csv("data/processed/features.csv", index=False)

    metrics = evaluate_time_cv(
        df_feat=df_feat,
        model_params=cfg["model"]["params"],
        cv_cfg=cfg["cv"],
    )
    save_metrics(metrics, cfg["output"]["metrics_path"])
    
    os.makedirs("outputs/forecasts", exist_ok=True)

    last_fold_csv = "outputs/forecasts/last_fold_predictions.csv"
    save_last_fold_predictions(
        df_feat=df_feat,
        model_params=cfg["model"]["params"],
        cv_cfg=cfg["cv"],
        out_path=last_fold_csv
    )

    plot_last_fold(
        pred_csv_path=last_fold_csv,
        out_png_path="outputs/figures/last_fold_forecast_vs_actual.png"
    )
    
    # Final model trained on all features
    model, feature_cols = train_final_model(df_feat, cfg["model"]["params"])

    # Forecast next N days using original KPI history (date,y)
    future = recursive_forecast(
        history=kpi[["date", "y"]],
        model=model,
        lags=cfg["features"]["lags"],
        rolling_windows=cfg["features"]["rolling_windows"],
        horizon_days=cfg["cv"]["horizon_days"],
    )

    future.to_csv(cfg["output"]["forecast_path"], index=False)

    # Quick plot last 90 days (sanity check)
    tail = kpi.tail(90)
    plt.figure()
    plt.plot(tail["date"], tail["y"])
    plt.title("Daily revenue (last 90 days)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(cfg["output"]["figure_path"], dpi=150)

if __name__ == "__main__":
    main()