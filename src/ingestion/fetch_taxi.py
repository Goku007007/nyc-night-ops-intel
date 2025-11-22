# src/ingestion/fetch_taxi.py
from pathlib import Path
import requests
from typing import List
import sys

from src.config import RAW_TAXI_DIR, TLC_TAXI_BASE_URL
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def build_taxi_url(year: int, month: int) -> str:
    return f"{TLC_TAXI_BASE_URL}/yellow_tripdata_{year}-{month:02d}.parquet"

def download_taxi_month(year: int, month: int, overwrite: bool = False) -> Path:
    RAW_TAXI_DIR.mkdir(parents=True, exist_ok=True)
    fname = RAW_TAXI_DIR / f"yellow_tripdata_{year}-{month:02d}.parquet"
    if fname.exists() and not overwrite:
        logger.info(f"Taxi file already exists, skipping: {fname}")
        return fname

    url = build_taxi_url(year, month)
    logger.info(f"Downloading taxi data from {url}")

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    with open(fname, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    logger.info(f"Saved taxi data to {fname}")
    return fname

def download_taxi_range(year: int, months: List[int]) -> None:
    for m in months:
        try:
            download_taxi_month(year, m)
        except Exception as e:
            logger.error(f"Failed for {year}-{m:02d}: {e}", exc_info=True)

if __name__ == "__main__":
    # Usage: python -m src.ingestion.fetch_taxi 2023 10 11 12
    if len(sys.argv) < 3:
        print("Usage: python -m src.ingestion.fetch_taxi YEAR M1 M2 ...")
        sys.exit(1)
    year = int(sys.argv[1])
    months = [int(x) for x in sys.argv[2:]]
    download_taxi_range(year, months)
