"""Central place for paths and column names. Change names here, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "APL_Logistics.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "apl_clean.parquet"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
ENCODING = "latin-1"          # the file is not UTF-8

# Core columns
TARGET = "Late_delivery_risk"
REAL_DAYS = "Days for shipping (real)"
SCHED_DAYS = "Days for shipment (scheduled)"
DELIVERY_STATUS = "Delivery Status"
SHIP_MODE = "Shipping Mode"
MARKET = "Market"
REGION = "Order Region"
COUNTRY = "Order Country"
SEGMENT = "Customer Segment"

# Columns created by cleaning
GAP = "delay_gap"
DELIVERY_CLASS = "delivery_class"
IS_CANCELED = "is_canceled"

# Personal data (not needed for any analysis) and a duplicate ID column
PII_COLS = ["Customer Fname", "Customer Lname", "Customer Street", "Customer Zipcode"]
REDUNDANT_COLS = ["Order Customer Id"]   # identical to Customer Id in this file

VALID_SCHEDULED_DAYS = {0, 1, 2, 4}
