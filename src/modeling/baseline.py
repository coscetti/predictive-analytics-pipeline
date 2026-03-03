import numpy as np
import pandas as pd

def seasonal_naive(y: pd.Series, season: int = 7) -> np.ndarray:
    """Seasonal naive baseline: y_hat[t] = y[t-season]."""
    return y.shift(season).to_numpy()