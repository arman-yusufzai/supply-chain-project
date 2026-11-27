"""Cost-based decision layer.

Framing: flagging an order costs c_int (an intervention: upgrade, expedite,
proactive notice). An intervention prevents lateness with probability
`effectiveness`. A late order that is not prevented costs c_late (SLA penalty,
unhappy customer). These are ASSUMPTIONS, not figures from the data.

Flag an order when  effectiveness * P(late) * c_late > c_int,
i.e. when P(late) > c_int / (effectiveness * c_late).
"""
import numpy as np
import pandas as pd


def break_even_probability(cost_ratio: float, effectiveness: float = 1.0) -> float:
    """cost_ratio = c_late / c_int."""
    return 1.0 / (cost_ratio * effectiveness)


def policy_result(y, flagged, cost_ratio: float, effectiveness: float = 1.0) -> dict:
    """Net saving versus doing nothing, in units of one intervention cost,
    per 1,000 orders, using the real outcomes y (1 = late)."""
    y = np.asarray(y); flagged = np.asarray(flagged).astype(bool)
    saving = (effectiveness * y[flagged] * cost_ratio - 1.0).sum()
    return {
        "share_flagged": flagged.mean(),
        "recall": (y[flagged].sum() / y.sum()) if y.sum() else np.nan,
        "precision": y[flagged].mean() if flagged.any() else np.nan,
        "saving_per_1000": 1000 * saving / len(y),
    }


def policy_grid(y, p, ratios=(2, 3, 5, 10), efficacies=(0.3, 0.5, 1.0)) -> pd.DataFrame:
    """Best flagging rule for each (cost ratio, effectiveness) pair."""
    p = np.asarray(p)
    rows = []
    for r in ratios:
        for e in efficacies:
            t = break_even_probability(r, e)
            res = policy_result(y, p > t, r, e)
            all_res = policy_result(y, np.ones(len(p), bool), r, e)
            rows.append({"cost_ratio": r, "effectiveness": e,
                         "threshold": round(t, 3), **res,
                         "saving_flag_all": all_res["saving_per_1000"]})
    return pd.DataFrame(rows)
