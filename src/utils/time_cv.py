from dataclasses import dataclass
import pandas as pd

@dataclass
class Fold:
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

def rolling_folds(dates: pd.Series, initial_train_days: int, horizon_days: int, step_days: int, n_folds: int):
    dates = pd.to_datetime(dates).sort_values().unique()
    start = dates[0]
    folds = []
    train_end = start + pd.Timedelta(days=initial_train_days)

    for _ in range(n_folds):
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=horizon_days - 1)
        folds.append(Fold(train_end=train_end, test_start=test_start, test_end=test_end))
        train_end = train_end + pd.Timedelta(days=step_days)

    return folds