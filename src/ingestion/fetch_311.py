# src/ingestion/fetch_311.py
from datetime import datetime, timedelta
from pathlib import Path
import os
import sys
from typing import List, Dict, Any

import pandas as pd
import requests

from src.config import RAW_311_DIR, NYC_311_APP_TOKEN
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

BASE_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

def build_where_clause(start_date: str, end_date: str) -> str:
    """
    Build a SoQL $where clause for the NYC 311 dataset.

    Notes:
    - SoQL does NOT support ILIKE. For case-insensitive search we use:
      upper(complaint_type) like '%NOISE%'
    - start_date and end_date are YYYY-MM-DD (inclusive start, exclusive end).
    """
    complaint_filter = (
        "upper(complaint_type) like '%NOISE%' OR "
        "upper(complaint_type) like '%DISORDERLY%' OR "
        "upper(complaint_type) like '%ASSAULT%' OR "
        "upper(complaint_type) like '%LOUD MUSIC%'"
    )

    date_filter = (
        f"created_date >= '{start_date}T00:00:00' AND "
        f"created_date < '{end_date}T00:00:00'"
    )

    # You can also later add a time-of-day filter (20:00–06:00) using
    # date_extract_hh(created_date) between 20 and 23 OR ... etc.
    return f"({complaint_filter}) AND ({date_filter})"


def fetch_page(
    where_clause: str,
    limit: int = 50000,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    headers = {}
    if NYC_311_APP_TOKEN:
        headers["X-App-Token"] = NYC_311_APP_TOKEN

    params = {
        "$where": where_clause,
        "$limit": limit,
        "$offset": offset,
        "$order": "created_date ASC",
        "$select": ",".join(
            [
                "unique_key",
                "created_date",
                "closed_date",
                "complaint_type",
                "descriptor",
                "agency",
                "borough",
                "incident_zip",
                "latitude",
                "longitude",
                "location_type",
            ]
        ),
    }

    logger.info(f"Requesting 311 page offset={offset}")
    resp = requests.get(BASE_URL, params=params, headers=headers, timeout=60)

    if not resp.ok:
        # Show the server's error message to help debug SoQL issues
        logger.error(
            "311 API error %s: %s",
            resp.status_code,
            resp.text,
        )
        resp.raise_for_status()

    return resp.json()


def download_311_period(
    start_date: str,
    end_date: str,
    output_path: Path,
    page_limit: int = 50000,
) -> Path:
    RAW_311_DIR.mkdir(parents=True, exist_ok=True)
    where_clause = build_where_clause(start_date, end_date)

    all_rows: List[Dict[str, Any]] = []
    offset = 0

    while True:
        page = fetch_page(where_clause, limit=page_limit, offset=offset)
        if not page:
            break
        all_rows.extend(page)
        offset += page_limit
        logger.info(f"Fetched {len(page)} rows, total={len(all_rows)}")

        # Safety: stop if too big in MVP
        if offset >= 300000:
            logger.warning("Stopping after 300k rows for MVP")
            break

    if not all_rows:
        logger.warning("No 311 rows fetched for given period")
        return output_path

    df = pd.DataFrame(all_rows)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved 311 data to {output_path}")
    return output_path

if __name__ == "__main__":
    # Usage: python -m src.ingestion.fetch_311 2023-10-01 2024-03-31
    if len(sys.argv) != 3:
        print("Usage: python -m src.ingestion.fetch_311 START_DATE END_DATE (YYYY-MM-DD)")
        sys.exit(1)

    start_date = sys.argv[1]
    end_date = sys.argv[2]
    out = RAW_311_DIR / f"311_night_{start_date}_to_{end_date}.parquet"
    download_311_period(start_date, end_date, out)
