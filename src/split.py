"""Train/test split. The test set is created once and not used until the end."""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src import config

GROUP_COL = "Customer Id"


def make_split(d: pd.DataFrame, test_fraction: float = 0.2):
    """Stratified on the target, grouped by customer.

    Grouped: no customer appears in both train and test, so the test score
    is not inflated by seeing the same customer twice.
    """
    n_splits = round(1 / test_fraction)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(d, d[config.TARGET], groups=d[GROUP_COL]))
    return d.iloc[train_idx].reset_index(drop=True), d.iloc[test_idx].reset_index(drop=True)
