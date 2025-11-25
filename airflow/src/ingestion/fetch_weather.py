# src/ingestion/fetch_weather.py

import sys
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd

from src.config import RAW_WEATHER_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/era5"

# Approx NYC (Manhattan)
LATITUDE = 40.7128
LONGITUDE = -74.0060
TIMEZONE = "America/New_York"


def parse_date(d: str) -> str:
    """Validate and return date string in YYYY-MM-DD."""
    try:
        datetime.strptime(d, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format: {d}. Use YYYY-MM-DD.")
    return d


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly historical weather for NYC from Open-Meteo.
    Returns a DataFrame with time, temperature, precipitation, snowfall, weather code.
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation,snowfall,weathercode",
        "timezone": TIMEZONE,
    }

    logger.info(f"Requesting weather from {OPEN_METEO_URL} for {start_date} to {end_date}")
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precips = hourly.get("precipitation", [])
    snows = hourly.get("snowfall", [])
    codes = hourly.get("weathercode", [])

    if not times:
        raise RuntimeError("No hourly weather data returned")

    if not (len(times) == len(temps) == len(precips) == len(snows) == len(codes)):
        raise RuntimeError("Inconsistent lengths in hourly weather arrays")

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(times),
            "temperature_2m": temps,
            "precipitation": precips,
            "snowfall": snows,
            "weathercode": codes,
        }
    )

    return df


def save_weather(df: pd.DataFrame, start_date: str, end_date: str) -> Path:
    RAW_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_WEATHER_DIR / f"weather_{start_date}_{end_date}.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"Wrote weather parquet to {out}")
    return out


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m src.ingestion.fetch_weather START_DATE END_DATE")
        print("Example: python -m src.ingestion.fetch_weather 2023-10-01 2024-01-01")
        sys.exit(1)

    start = parse_date(sys.argv[1])
    end = parse_date(sys.argv[2])

    df = fetch_weather(start, end)
    save_weather(df, start, end)


if __name__ == "__main__":
    main()
