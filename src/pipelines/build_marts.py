# src/pipelines/build_marts.py
from pathlib import Path
import duckdb
import pandas as pd

from src.config import (
    RAW_TAXI_DIR,
    RAW_311_DIR,
    SILVER_DIR,
    GOLD_DIR,
    BASE_DIR,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

DB_PATH = BASE_DIR / "data" / "night_ops.duckdb"

def build_silver_tables() -> None:
    logger.info("Building silver tables in DuckDB")
    conn = duckdb.connect(DB_PATH.as_posix())

    # Attach taxi Parquet files as one virtual table
    taxi_files = sorted(RAW_TAXI_DIR.glob("yellow_tripdata_*.parquet"))
    if not taxi_files:
        raise RuntimeError("No taxi files found in RAW_TAXI_DIR")

    taxi_glob = str(RAW_TAXI_DIR / "yellow_tripdata_*.parquet")
    conn.execute(
        f"""
        CREATE OR REPLACE VIEW taxi_raw AS
        SELECT
            tpep_pickup_datetime,
            tpep_dropoff_datetime,
            PULocationID,
            DOLocationID,
            passenger_count,
            trip_distance,
            fare_amount,
            total_amount
        FROM read_parquet('{taxi_glob}');
        """
    )

    # Hourly taxi pickups by pickup zone
    conn.execute(
        """
        CREATE OR REPLACE TABLE silver_taxi_hourly AS
        SELECT
            date_trunc('hour', tpep_pickup_datetime) AS pickup_hour,
            PULocationID AS zone_id,
            COUNT(*) AS trips,
            AVG(trip_distance) AS avg_trip_distance,
            AVG(total_amount) AS avg_total_amount,
            SUM(total_amount) AS revenue
        FROM taxi_raw
        WHERE trip_distance > 0
          AND total_amount > 0
        GROUP BY 1, 2;
        """
    )

    # 311
    complaint_files = sorted(RAW_311_DIR.glob("311_night_*.parquet"))
    if not complaint_files:
        raise RuntimeError("No 311 files found in RAW_311_DIR")

    complaints_glob = str(RAW_311_DIR / "311_night_*.parquet")
    conn.execute(
        f"""
        CREATE OR REPLACE VIEW complaints_raw AS
        SELECT
            created_date::TIMESTAMP AS created_date,
            complaint_type,
            descriptor,
            borough,
            incident_zip,
            latitude,
            longitude
        FROM read_parquet('{complaints_glob}');
        """
    )

    conn.execute(
        """
        CREATE OR REPLACE TABLE silver_311_hourly AS
        SELECT
            date_trunc('hour', created_date) AS complaint_hour,
            borough,
            incident_zip,
            complaint_type,
            COUNT(*) AS complaints
        FROM complaints_raw
        GROUP BY 1, 2, 3, 4;
        """
    )

    logger.info("Silver tables created: silver_taxi_hourly, silver_311_hourly")
    conn.close()

def build_gold_features() -> None:
    logger.info("Building gold features table")
    conn = duckdb.connect(DB_PATH.as_posix())

    # Join taxi to 311 using time + borough-ish proxy via ZIP/zone.
    # MVP: approximate borough using taxi zone lookup later; for now
    # we aggregate 311 by hour only and join on hour.

    conn.execute(
        """
        CREATE OR REPLACE TABLE gold_hourly_features AS
        WITH complaints_by_hour AS (
            SELECT
                complaint_hour AS hour,
                COUNT(*) AS complaints_total
            FROM silver_311_hourly
            GROUP BY 1
        )
        SELECT
            t.pickup_hour AS hour,
            t.zone_id,
            t.trips,
            t.avg_trip_distance,
            t.avg_total_amount,
            t.revenue,
            COALESCE(c.complaints_total, 0) AS complaints_total
        FROM silver_taxi_hourly t
        LEFT JOIN complaints_by_hour c
          ON t.pickup_hour = c.hour;
        """
    )

    # Export for Tableau
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    df_gold = conn.execute("SELECT * FROM gold_hourly_features").fetch_df()
    out_csv = GOLD_DIR / "gold_hourly_features.csv"
    df_gold.to_csv(out_csv, index=False)
    logger.info(f"Wrote gold features CSV for Tableau: {out_csv}")

    conn.close()

if __name__ == "__main__":
    build_silver_tables()
    build_gold_features()
