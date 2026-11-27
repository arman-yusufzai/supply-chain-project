"""Final test-set evaluation helpers. Used once, at the very end."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def cluster_bootstrap_ci(y, p, groups, metric, n_boot: int = 300, seed: int = 42):
    """95% interval for a metric, resampling whole customers (not single rows),
    because a customer's orders are not independent of each other."""
    y, p = np.asarray(y), np.asarray(p)
    idx = list(pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).indices.values())
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx), len(idx))
        rows = np.concatenate([idx[i] for i in pick])
        vals.append(metric(y[rows], p[rows]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _logloss(y, p):
    return log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))


def scorecard_with_ci(y, p, groups, n_boot: int = 300) -> pd.DataFrame:
    rows = {}
    for name, fn in [("roc_auc", roc_auc_score), ("log_loss", _logloss),
                     ("brier", brier_score_loss)]:
        lo, hi = cluster_bootstrap_ci(y, p, groups, fn, n_boot)
        rows[name] = {"estimate": fn(np.asarray(y), np.asarray(p)),
                      "ci_low": lo, "ci_high": hi}
    return pd.DataFrame(rows).T.round(4)
