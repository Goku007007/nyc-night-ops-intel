# src/config.py
from pathlib import Path
from dotenv import load_dotenv
import os

# Project root (airflow/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env
load_dotenv(BASE_DIR / ".env", override=False)

DATA_DIR = BASE_DIR / "data"

# Raw data dirs
RAW_TAXI_DIR = DATA_DIR / "raw" / "taxi"
RAW_311_DIR = DATA_DIR / "raw" / "311"
RAW_REFERENCE_DIR = DATA_DIR / "raw" / "reference"  # for taxi_zone_lookup.csv
RAW_WEATHER_DIR = DATA_DIR / "raw" / "weather"      # for weather_YYYY-MM-DD_YYYY-MM-DD.parquet

# Processed data dirs
SILVER_DIR = DATA_DIR / "processed" / "silver"
GOLD_DIR = DATA_DIR / "processed" / "gold"

# Ensure directories exist
for d in [RAW_TAXI_DIR, RAW_311_DIR, RAW_REFERENCE_DIR, RAW_WEATHER_DIR, SILVER_DIR, GOLD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# External configs
TLC_TAXI_BASE_URL = os.getenv(
    "TLC_TAXI_BASE_URL",
    "https://d37ci6vzurychx.cloudfront.net/trip-data",
)
NYC_311_APP_TOKEN = os.getenv("NYC_311_APP_TOKEN")
