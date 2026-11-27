"""Model factories. Every model is a full Pipeline: preprocessing + estimator."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from src import config
from src.features import build_preprocessor

# Feature groups, used to ask "does this group add anything beyond shipping mode?"
FEATURE_GROUPS = {
    "geography": ["Market", "Order Region", "Order Country", "Customer State"],
    "customer": ["Customer Segment", "Type", "Customer Country"],
    "product": ["Category Name", "Department Name", "Product Name"],
    "order_value": ["Order Item Discount", "Order Item Discount Rate",
                    "Order Item Product Price", "Order Item Quantity",
                    "Order Item Total"],
    "coordinates": ["Latitude", "Longitude"],
}


def _split(cols, numeric):
    num = [c for c in cols if c in numeric]
    cat = [c for c in cols if c not in numeric]
    return num, cat


def make_logreg(cols, numeric, C: float = 1.0) -> Pipeline:
    num, cat = _split(cols, numeric)
    return Pipeline([("prep", build_preprocessor(num, cat)),
                     ("clf", LogisticRegression(C=C, max_iter=2000))])


def make_hgb(cols, numeric, **params) -> Pipeline:
    """Gradient boosting with integer-coded categories (unseen -> missing)."""
    num, cat = _split(cols, numeric)
    prep = ColumnTransformer(
        [("num", "passthrough", num),
         ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=np.nan,
                                encoded_missing_value=np.nan), cat)],
        verbose_feature_names_out=False)
    mask = [False] * len(num) + [True] * len(cat)
    clf = HistGradientBoostingClassifier(categorical_features=mask,
                                         random_state=config.RANDOM_STATE, **params)
    return Pipeline([("prep", prep), ("clf", clf)])
