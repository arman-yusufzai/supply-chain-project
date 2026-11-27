"""Metrics and cross-validation helpers. Everything here works on TRAIN data only."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from src import config


def make_cv(n_splits: int = 5):
    """Same grouped + stratified logic as the train/test split."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=config.RANDOM_STATE)


def oof_predictions(model, X, y, groups, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold P(late): each row is predicted by a model that never saw it
    (or any order of the same customer)."""
    oof = np.zeros(len(y))
    for tr, va in make_cv(n_splits).split(X, y, groups):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def _pr_auc(y, p) -> float:
    """PR-AUC with ties broken by tiny seeded noise.

    A model with few distinct scores (e.g. a shipping-mode lookup) has huge tied
    blocks, and sklearn scores a tied block as one lump, understating PR-AUC.
    Breaking ties the same way for every model keeps comparisons fair.
    """
    rng = np.random.default_rng(config.RANDOM_STATE)
    return average_precision_score(y, np.asarray(p) + rng.normal(0, 1e-7, len(p)))


def score_row(y, p, threshold: float = 0.5) -> dict:
    """Standard scorecard for one set of predicted probabilities."""
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": _pr_auc(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "brier": brier_score_loss(y, p),
        "accuracy@0.5": accuracy_score(y, pred),
        "precision@0.5": precision_score(y, pred, zero_division=0),
        "recall@0.5": recall_score(y, pred, zero_division=0),
    }


def fold_scores(model, X, y, groups, n_splits: int = 5) -> pd.DataFrame:
    """Score per fold, to see how stable a result is."""
    rows = []
    for i, (tr, va) in enumerate(make_cv(n_splits).split(X, y, groups), 1):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        rows.append({"fold": i, **score_row(y.iloc[va], p)})
    return pd.DataFrame(rows).set_index("fold")
