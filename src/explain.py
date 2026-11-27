"""Lift/gains tables and SHAP helpers."""
import numpy as np
import pandas as pd


def gains_by_group(y, p, labels) -> pd.DataFrame:
    """Gains and lift when orders are ranked by group risk (a few distinct scores).

    Within a group every order has the same score, so the gains curve is a
    straight line across that group.
    """
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "g": np.asarray(labels)})
    t = (df.groupby("g").agg(orders=("y", "size"), late=("y", "sum"), score=("p", "mean"))
           .sort_values("score", ascending=False))
    base = df["y"].mean()
    t["late_rate"] = t["late"] / t["orders"]
    t["lift"] = t["late_rate"] / base
    t["cum_orders_share"] = t["orders"].cumsum() / len(df)
    t["cum_recall"] = t["late"].cumsum() / df["y"].sum()
    return t.drop(columns="score")


def gains_by_decile(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Gains and lift for a continuous score, in equal-sized bins, highest risk first."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df = df.sort_values("p", ascending=False, kind="mergesort").reset_index(drop=True)
    df["bin"] = (np.arange(len(df)) * n_bins // len(df)) + 1
    t = df.groupby("bin").agg(orders=("y", "size"), late=("y", "sum"))
    t["late_rate"] = t["late"] / t["orders"]
    t["lift"] = t["late_rate"] / df["y"].mean()
    t["cum_orders_share"] = t["orders"].cumsum() / len(df)
    t["cum_recall"] = t["late"].cumsum() / df["y"].sum()
    return t


def mean_abs_shap(shap_values, feature_names) -> pd.Series:
    return (pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names)
              .sort_values(ascending=False))
