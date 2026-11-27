"""KPI and diagnostic helpers. Reused by notebooks and the Streamlit app."""
import numpy as np
import pandas as pd
from src import config


def group_summary(d: pd.DataFrame, by: str) -> pd.DataFrame:
    """Orders, late rate, on-time rate and average delay gap per group."""
    g = d.groupby(by).agg(
        orders=(config.GAP, "size"),
        late_rate=(config.TARGET, "mean"),
        avg_gap=(config.GAP, "mean"),
    )
    g["on_time_rate"] = 1 - g["late_rate"]
    return g.round(4)


def mode_adjusted(d: pd.DataFrame, by: str) -> pd.DataFrame:
    """Is a group later than its shipping-mode mix would predict?

    expected_rate = late rate the group would have if each of its orders were as
    late as the average order of the same shipping mode.
    z = (observed - expected) / standard error. |z| above ~3 is hard to explain
    by chance; with many groups a few |z| near 2 are normal.
    """
    p = d.groupby(config.SHIP_MODE)[config.TARGET].transform("mean")
    t = d.assign(_p=p, _v=p * (1 - p))
    g = t.groupby(by).agg(orders=(config.TARGET, "size"),
                          obs=(config.TARGET, "sum"),
                          exp=("_p", "sum"), var=("_v", "sum"))
    out = pd.DataFrame({
        "orders": g["orders"],
        "observed_rate": g["obs"] / g["orders"],
        "expected_rate": g["exp"] / g["orders"],
        "z": (g["obs"] - g["exp"]) / np.sqrt(g["var"]),
    })
    return out.round(4).sort_values("z")
