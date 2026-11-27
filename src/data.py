"""Data loading and cleaning."""
import pandas as pd
from src import config


def load_raw() -> pd.DataFrame:
    """Read the raw CSV exactly as delivered, with no changes."""
    return pd.read_csv(config.RAW_PATH, encoding=config.ENCODING)


def clean(df_raw: pd.DataFrame):
    """Return (clean_df, report). Never modifies df_raw."""
    df = df_raw.copy()
    report = {"rows_in": len(df)}

    # 1. Standardise text: trim and collapse repeated spaces
    #    (e.g. 'South of  USA ' -> 'South of USA')
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in text_cols:
        df[c] = df[c].str.strip().str.replace(r"\s+", " ", regex=True)

    # 2. Validate shipping durations
    bad_real = df[config.REAL_DAYS].isna() | (df[config.REAL_DAYS] < 0)
    bad_sched = ~df[config.SCHED_DAYS].isin(config.VALID_SCHEDULED_DAYS)
    report["invalid_rows_dropped"] = int((bad_real | bad_sched).sum())
    df = df[~(bad_real | bad_sched)]

    # 3. Remove personal data and the duplicate ID column
    df = df.drop(columns=config.PII_COLS + config.REDUNDANT_COLS, errors="ignore")

    # 4. Delay gap and delivery class
    df[config.GAP] = df[config.REAL_DAYS] - df[config.SCHED_DAYS]
    df[config.DELIVERY_CLASS] = pd.cut(
        df[config.GAP], bins=[-99, -1, 0, 99],
        labels=["Early", "On-time", "Delayed"],
    ).astype("object")

    # 5. Canceled shipments never reached the customer: flag, do not delete
    df[config.IS_CANCELED] = df[config.DELIVERY_STATUS].eq("Shipping canceled")

    report["rows_out"] = len(df)
    report["canceled_flagged"] = int(df[config.IS_CANCELED].sum())
    return df.reset_index(drop=True), report


def delivered_only(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that actually shipped. Use this for every delay analysis."""
    return df[~df[config.IS_CANCELED]]
