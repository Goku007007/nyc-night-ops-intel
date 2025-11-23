# src/pipelines/train_models.py
import duckdb
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import joblib

from src.config import BASE_DIR, GOLD_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)
DB_PATH = BASE_DIR / "data" / "night_ops.duckdb"

def load_gold() -> pd.DataFrame:
    conn = duckdb.connect(DB_PATH.as_posix())
    df = conn.execute(
        """
        SELECT
            hour,
            zone_id,
            trips,
            avg_trip_distance,
            avg_total_amount,
            revenue,
            complaints_total
        FROM gold_hourly_features
        """
    ).fetch_df()
    conn.close()
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df

def train_and_save_models() -> None:
    df = load_gold()
    df = add_time_features(df)

    feature_cols = [
        "zone_id",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "avg_trip_distance",
        "avg_total_amount",
        "revenue",
        "complaints_total",
    ]

    # Drop rows with nulls in features/targets
    df = df.dropna(subset=feature_cols + ["trips"])

    X = df[feature_cols]
    y_trips = df["trips"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_trips, test_size=0.2, random_state=42
    )

    model_trips = GradientBoostingRegressor(random_state=42)
    model_trips.fit(X_train, y_train)

    y_pred = model_trips.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"Trips model MAE: {mae:.2f}")

    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model_trips, models_dir / "gb_trips.pkl")

    # Complaints model using similar features (but exclude complaints_total as feature)
    df2 = df.dropna(subset=["complaints_total"])
    X2 = df2[[
        "zone_id",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "avg_trip_distance",
        "avg_total_amount",
        "revenue",
    ]]
    y_complaints = df2["complaints_total"]

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y_complaints, test_size=0.2, random_state=42
    )

    model_complaints = GradientBoostingRegressor(random_state=42)
    model_complaints.fit(X2_train, y2_train)

    y2_pred = model_complaints.predict(X2_test)
    mae2 = mean_absolute_error(y2_test, y2_pred)
    logger.info(f"Complaints model MAE: {mae2:.2f}")

    joblib.dump(model_complaints, models_dir / "gb_complaints.pkl")

    # Optionally write predictions back to Gold for Tableau
    df_scores = df.copy()
    df_scores["pred_trips"] = model_trips.predict(df_scores[feature_cols])
    df_scores["pred_complaints"] = model_complaints.predict(
        df_scores[[
            "zone_id",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "avg_trip_distance",
            "avg_total_amount",
            "revenue",
        ]]
    )

    out_csv = GOLD_DIR / "gold_hourly_with_predictions.csv"
    df_scores.to_csv(out_csv, index=False)
    logger.info(f"Wrote predictions to {out_csv}")

if __name__ == "__main__":
    train_and_save_models()
