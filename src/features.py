"""Which columns may be used to predict lateness, and the preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import config

# Known only after the shipment happens, or computed from the outcome itself
OUTCOME_COLS = [
    config.REAL_DAYS, config.GAP, config.DELIVERY_STATUS,
    config.DELIVERY_CLASS, config.IS_CANCELED,
]

# Not known at order time, or net of costs that arise during fulfilment.
# Excluded to be safe; can be revisited if the business confirms timing.
EXCLUDED_UNSURE_COLS = [
    "Order Status",                      # changes after the order is placed
    "Benefit per order", "Order Profit Per Order", "Order Item Profit Ratio",
]

# Duplicates of other information (each checked against the data)
REDUNDANT_COLS = [
    config.SCHED_DAYS,                   # fully determined by Shipping Mode
    "Category Id", "Department Id",      # same as the name columns
    "Sales per customer",                # equals Order Item Total on 100% of rows
    "Product Price",                     # equals Order Item Product Price on 100% of rows
    "Sales",                             # equals price x quantity on 100% of rows
]

# Identifiers (a model should not memorise individuals)
ID_COLS = ["Customer Id"]

# Too many levels for a sparse, mostly-small-group column (3,488 / 1,062 / 563
# levels in train). Geography is still covered by Market, Region, Country, State.
HIGH_CARDINALITY_DROP = ["Order City", "Order State", "Customer City"]

NEVER_USE = ([config.TARGET] + OUTCOME_COLS + EXCLUDED_UNSURE_COLS
             + REDUNDANT_COLS + ID_COLS + HIGH_CARDINALITY_DROP)

MIN_CATEGORY_COUNT = 100   # categories rarer than this are pooled into "infrequent"


def candidate_features(df: pd.DataFrame) -> list:
    """Columns left after removing everything above."""
    return [c for c in df.columns if c not in NEVER_USE]


def split_feature_types(df: pd.DataFrame):
    feats = candidate_features(df)
    numeric = df[feats].select_dtypes(include="number").columns.tolist()
    categorical = [c for c in feats if c not in numeric]
    return numeric, categorical


def build_preprocessor(numeric, categorical) -> ColumnTransformer:
    """Imputers, scalers and encoders. Fitted only inside a Pipeline, so in
    cross-validation they learn from the training folds alone."""
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=MIN_CATEGORY_COUNT)),
    ])
    return ColumnTransformer([("num", num_pipe, numeric),
                              ("cat", cat_pipe, categorical)])
