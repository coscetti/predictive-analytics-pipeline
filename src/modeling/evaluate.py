import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.modeling.baseline import seasonal_naive
from src.utils.time_cv import rolling_folds

def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(y_true == 0, 1.0, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def evaluate_time_cv(df_feat: pd.DataFrame, model_params: dict, cv_cfg: dict) -> dict:
    folds = rolling_folds(
        dates=df_feat["date"],
        initial_train_days=cv_cfg["initial_train_days"],
        horizon_days=cv_cfg["horizon_days"],
        step_days=cv_cfg["step_days"],
        n_folds=cv_cfg["n_folds"],
    )

    feature_cols = [c for c in df_feat.columns if c not in ("date", "y")]
    results = {"folds": []}

    for i, fold in enumerate(folds, start=1):
        train = df_feat[df_feat["date"] <= fold.train_end]
        test = df_feat[(df_feat["date"] >= fold.test_start) & (df_feat["date"] <= fold.test_end)]

        if len(test) == 0 or len(train) < 30:
            continue

        # Baseline
        base_pred_all = seasonal_naive(df_feat.set_index("date")["y"], season=7)
        test_idx = test.index.to_numpy()
        base_pred = base_pred_all[test_idx]

        # XGB
        model = XGBRegressor(**model_params)
        model.fit(train[feature_cols], train["y"])
        pred = model.predict(test[feature_cols])

        mse_base = mean_squared_error(test["y"], base_pred)
        mse_xgb = mean_squared_error(test["y"], pred)

        fold_metrics = {
            "fold": i,
            "train_end": str(fold.train_end.date()),
            "test_start": str(fold.test_start.date()),
            "test_end": str(fold.test_end.date()),
            "baseline": {
                "mae": float(mean_absolute_error(test["y"], base_pred)),
                "rmse": float(np.sqrt(mse_base)),
                "mape": float(mape(test["y"], base_pred)),
            },
            "xgboost": {
                "mae": float(mean_absolute_error(test["y"], pred)),
                "rmse": float(np.sqrt(mse_xgb)),
                "mape": float(mape(test["y"], pred)),
            },
        }
        results["folds"].append(fold_metrics)

    # Aggregate
    def agg(model_key, metric):
        vals = [f[model_key][metric] for f in results["folds"]]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else {"mean": None, "std": None}

    results["aggregate"] = {
        "baseline": {m: agg("baseline", m) for m in ["mae", "rmse", "mape"]},
        "xgboost": {m: agg("xgboost", m) for m in ["mae", "rmse", "mape"]},
    }
    return results

def save_metrics(metrics: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

def save_last_fold_predictions(
    df_feat: pd.DataFrame,
    model_params: dict,
    cv_cfg: dict,
    out_path: str
) -> str:
    """
    Runs the same rolling CV logic but saves predictions for the last valid fold:
    actual y, baseline yhat, xgboost yhat.
    """
    folds = rolling_folds(
        dates=df_feat["date"],
        initial_train_days=cv_cfg["initial_train_days"],
        horizon_days=cv_cfg["horizon_days"],
        step_days=cv_cfg["step_days"],
        n_folds=cv_cfg["n_folds"],
    )

    feature_cols = [c for c in df_feat.columns if c not in ("date", "y")]

    last_df = None

    # Precompute baseline series aligned with df_feat index
    base_pred_all = seasonal_naive(df_feat.set_index("date")["y"], season=7)

    for fold in folds:
        train = df_feat[df_feat["date"] <= fold.train_end]
        test = df_feat[(df_feat["date"] >= fold.test_start) & (df_feat["date"] <= fold.test_end)]
        if len(test) == 0 or len(train) < 30:
            continue

        # Baseline for test window
        test_idx = test.index.to_numpy()
        base_pred = base_pred_all[test_idx]

        # XGB
        model = XGBRegressor(**model_params)
        model.fit(train[feature_cols], train["y"])
        pred = model.predict(test[feature_cols])

        tmp = test[["date", "y"]].copy()
        tmp["yhat_baseline"] = base_pred
        tmp["yhat_xgboost"] = pred
        tmp["train_end"] = str(fold.train_end.date())
        tmp["test_start"] = str(fold.test_start.date())
        tmp["test_end"] = str(fold.test_end.date())

        last_df = tmp  # keep overwriting → last valid fold stays

    if last_df is None:
        raise RuntimeError("No valid fold found to save predictions.")

    last_df.to_csv(out_path, index=False)
    return out_path