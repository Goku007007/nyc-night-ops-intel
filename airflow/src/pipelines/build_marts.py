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

    # Ensure silver directory exists
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(DB_PATH.as_posix())

    # ----------------------------
    # Taxi: hourly pickups by zone
    # ----------------------------
    taxi_files = sorted(RAW_TAXI_DIR.glob("yellow_tripdata_*.parquet"))
    if not taxi_files:
        raise RuntimeError("No taxi files found in RAW_TAXI_DIR")

    taxi_glob = str(RAW_TAXI_DIR / "yellow_tripdata_*.parquet")
    logger.info(f"Reading taxi parquet files from {taxi_glob}")

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

    # ----------------------------
    # 311: hourly complaints
    # ----------------------------
    complaint_files = sorted(RAW_311_DIR.glob("311_night_*.parquet"))
    if not complaint_files:
        raise RuntimeError("No 311 files found in RAW_311_DIR")

    complaints_glob = str(RAW_311_DIR / "311_night_*.parquet")
    logger.info(f"Reading 311 parquet files from {complaints_glob}")

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

    # Borough + hour aggregation for complaints
    logger.info("Creating silver_311_borough_hourly")
    conn.execute(
        """
        CREATE OR REPLACE TABLE silver_311_borough_hourly AS
        SELECT
            complaint_hour,
            borough,
            SUM(complaints) AS borough_complaints_total
        FROM silver_311_hourly
        GROUP BY 1, 2;
        """
    )

    # ----------------------------
    # Taxi zone lookup enrichment
    # ----------------------------
    zone_lookup_path = (
        BASE_DIR / "data" / "raw" / "reference" / "taxi_zone_lookup.csv"
    )
    if not zone_lookup_path.exists():
        raise RuntimeError(
            f"TLC taxi zone lookup not found at {zone_lookup_path}. "
            "Download taxi_zone_lookup.csv and place it there."
        )

    logger.info(f"Creating silver_zone_lookup from {zone_lookup_path}")
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE silver_zone_lookup AS
        SELECT
            CAST(LocationID AS INTEGER)      AS zone_id,
            Borough                          AS borough,
            Zone                             AS zone_name,
            service_zone                     AS service_zone
        FROM read_csv_auto('{zone_lookup_path.as_posix()}');
        """
    )

    logger.info("Creating silver_taxi_hourly_enriched")
    conn.execute(
        """
        CREATE OR REPLACE TABLE silver_taxi_hourly_enriched AS
        SELECT
            t.pickup_hour,
            t.zone_id,
            z.borough,
            z.zone_name,
            z.service_zone,
            t.trips,
            t.avg_trip_distance,
            t.avg_total_amount,
            t.revenue
        FROM silver_taxi_hourly t
        LEFT JOIN silver_zone_lookup z
            ON t.zone_id = z.zone_id;
        """
    )

    logger.info(
        "Silver tables created: silver_taxi_hourly, silver_311_hourly, "
        "silver_311_borough_hourly, silver_zone_lookup, silver_taxi_hourly_enriched"
    )
    conn.close()


def build_gold_features() -> None:
    logger.info("Building gold features table")
    conn = duckdb.connect(DB_PATH.as_posix())

    # Base gold table: geography + borough-level complaints + intensity
    conn.execute(
        """
        CREATE OR REPLACE TABLE gold_hourly_features AS
        WITH complaints_by_borough_hour AS (
            SELECT
                complaint_hour AS hour,
                borough,
                borough_complaints_total
            FROM silver_311_borough_hourly
        )
        SELECT
            t.pickup_hour AS hour,
            t.zone_id,
            t.borough,
            t.zone_name,
            t.service_zone,
            t.trips,
            t.avg_trip_distance,
            t.avg_total_amount,
            t.revenue,
            COALESCE(c.borough_complaints_total, 0) AS borough_complaints_total,
            COALESCE(c.borough_complaints_total, 0) AS complaints_total,
            CASE
                WHEN t.trips > 0
                    THEN 100.0 * COALESCE(c.borough_complaints_total, 0) / t.trips
                ELSE NULL
            END AS complaints_per_100_trips
        FROM silver_taxi_hourly_enriched t
        LEFT JOIN complaints_by_borough_hour c
          ON t.pickup_hour = c.hour
         AND t.borough = c.borough
        WHERE t.pickup_hour >= TIMESTAMP '2023-10-01 00:00:00'
          AND t.pickup_hour <  TIMESTAMP '2024-01-01 00:00:00';
        """
    )

    # Add lag features using window functions
    logger.info("Adding lag features (lag1_trips, lag1_borough_complaints_total)")
    conn.execute(
        """
        CREATE OR REPLACE TABLE gold_hourly_features AS
        SELECT
            *,
            LAG(trips, 1) OVER (
                PARTITION BY zone_id
                ORDER BY hour
            ) AS lag1_trips,
            LAG(borough_complaints_total, 1) OVER (
                PARTITION BY borough
                ORDER BY hour
            ) AS lag1_borough_complaints_total
        FROM gold_hourly_features;
        """
    )

    # Export for Tableau / modeling
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    df_gold = conn.execute("SELECT * FROM gold_hourly_features").fetch_df()
    out_csv = GOLD_DIR / "gold_hourly_features.csv"
    df_gold.to_csv(out_csv, index=False)
    logger.info(f"Wrote gold features CSV for Tableau: {out_csv}")

    conn.close()



if __name__ == "__main__":
    build_silver_tables()
    build_gold_features()
