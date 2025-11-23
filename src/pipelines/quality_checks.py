# src/pipelines/quality_checks.py
import duckdb
from src.config import BASE_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)
DB_PATH = BASE_DIR / "data" / "night_ops.duckdb"

def check_gold_table() -> None:
    conn = duckdb.connect(DB_PATH.as_posix())

    df = conn.execute(
        "SELECT * FROM gold_hourly_features LIMIT 10000"
    ).fetch_df()
    conn.close()

    assert df["trips"].min() >= 0, "Negative trip counts found"
    assert df["revenue"].min() >= 0, "Negative revenue found"
    assert df["complaints_total"].min() >= 0, "Negative complaints found"

    # Basic null checks
    null_cols = [col for col in ["hour", "zone_id"] if df[col].isna().any()]
    if null_cols:
        raise AssertionError(f"Nulls in key columns: {null_cols}")

    logger.info("Quality checks passed for gold_hourly_features")

if __name__ == "__main__":
    check_gold_table()
