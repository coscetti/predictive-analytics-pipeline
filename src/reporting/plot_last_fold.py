import pandas as pd
import matplotlib.pyplot as plt

def plot_last_fold(pred_csv_path: str, out_png_path: str):
    df = pd.read_csv(pred_csv_path)
    df["date"] = pd.to_datetime(df["date"])

    plt.figure()
    plt.plot(df["date"], df["y"], label="actual")
    plt.plot(df["date"], df["yhat_baseline"], label="baseline")
    plt.plot(df["date"], df["yhat_xgboost"], label="xgboost")
    plt.title("Last fold: forecast vs actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_png_path, dpi=150)