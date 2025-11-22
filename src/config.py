# src/config.py
from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env
load_dotenv(BASE_DIR / ".env", override=False)

DATA_DIR = BASE_DIR / "data"
RAW_TAXI_DIR = DATA_DIR / "raw" / "taxi"
RAW_311_DIR = DATA_DIR / "raw" / "311"
SILVER_DIR = DATA_DIR / "processed" / "silver"
GOLD_DIR = DATA_DIR / "processed" / "gold"

RAW_TAXI_DIR.mkdir(parents=True, exist_ok=True)
RAW_311_DIR.mkdir(parents=True, exist_ok=True)
SILVER_DIR.mkdir(parents=True, exist_ok=True)
GOLD_DIR.mkdir(parents=True, exist_ok=True)

TLC_TAXI_BASE_URL = os.getenv(
    "TLC_TAXI_BASE_URL",
    "https://d37ci6vzurychx.cloudfront.net/trip-data",
)
NYC_311_APP_TOKEN = os.getenv("NYC_311_APP_TOKEN")
