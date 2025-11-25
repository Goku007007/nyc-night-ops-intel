# src/pipelines/train_models.py

from pathlib import Path
from datetime import date

import duckdb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import joblib

from src.config import BASE_DIR, GOLD_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)
DB_PATH = BASE_DIR / "data" / "night_ops.duckdb"


def load_gold() -> pd.DataFrame:
    """Load gold_hourly_features from DuckDB."""
    conn = duckdb.connect(DB_PATH.as_posix())
    df = conn.execute(
        """
        SELECT
            hour,
            zone_id,
            borough,
            zone_name,
            service_zone,
            trips,
            avg_trip_distance,
            avg_total_amount,
            revenue,
            borough_complaints_total,
            complaints_total,
            complaints_per_100_trips,
            lag1_trips,
            lag1_borough_complaints_total
        FROM gold_hourly_features
        """
    ).fetch_df()
    conn.close()
    return df


def add_time_and_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-of-day, day-of-week, and simple calendar/event flags."""
    df = df.copy()
    df["hour"] = pd.to_datetime(df["hour"])
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek  # 0 = Monday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Night flag for the core theme: 20:00–06:00
    df["is_night"] = ((df["hour_of_day"] >= 20) | (df["hour_of_day"] < 6)).astype(int)

    df["date"] = df["hour"].dt.date
    df["month"] = df["hour"].dt.month
    df["day"] = df["hour"].dt.day

    # Friday night: Fri 18:00–Sat 04:00
    df["is_friday_night"] = (
        ((df["day_of_week"] == 4) & (df["hour_of_day"] >= 18)) |
        ((df["day_of_week"] == 5) & (df["hour_of_day"] < 4))
    ).astype(int)

    # Simple US holiday list for Oct–Dec 2023 (your data window)
    holidays_2023 = {
        date(2023, 10, 9): "Columbus Day / Indigenous Peoples Day",
        date(2023, 11, 23): "Thanksgiving Day",
        date(2023, 12, 25): "Christmas Day",
    }

    df["holiday_name"] = df["date"].map(holidays_2023).fillna("")
    df["is_holiday"] = (df["holiday_name"] != "").astype(int)

    # New Year's Eve window: Dec 31 18:00–Jan 1 04:00
    df["is_nye"] = (
        ((df["month"] == 12) & (df["day"] == 31) & (df["hour_of_day"] >= 18)) |
        ((df["month"] == 1) & (df["day"] == 1) & (df["hour_of_day"] < 4))
    ).astype(int)

    # Halloween window: Oct 31 18:00–Nov 1 04:00
    df["is_halloween"] = (
        ((df["month"] == 10) & (df["day"] == 31) & (df["hour_of_day"] >= 18)) |
        ((df["month"] == 11) & (df["day"] == 1) & (df["hour_of_day"] < 4))
    ).astype(int)

    # Payday window: 1–3 and 15–17 of each month
    df["is_payday_window"] = df["day"].isin([1, 2, 3, 15, 16, 17]).astype(int)

    return df


def train_and_save_models() -> None:
    logger.info("Loading gold features for model training")
    df = load_gold()
    df = add_time_and_calendar_features(df)

    # -----------------------------------
    # Trips model (time-based train/test)
    # -----------------------------------
    feature_cols_trips = [
        "zone_id",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_friday_night",
        "is_holiday",
        "is_nye",
        "is_halloween",
        "is_payday_window",
        "is_night",
        "avg_trip_distance",
        "avg_total_amount",
        "borough_complaints_total",
        "lag1_trips",
        "lag1_borough_complaints_total",
    ]

    df_trips = df.dropna(subset=feature_cols_trips + ["trips"]).copy()
    df_trips = df_trips.sort_values("hour")

    X = df_trips[feature_cols_trips]
    y_trips = df_trips["trips"]

    split_idx = int(len(df_trips) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_trips.iloc[:split_idx], y_trips.iloc[split_idx:]

    logger.info(
        f"Trips model training on {len(X_train)} rows, testing on {len(X_test)} rows "
        f"(time-based split)"
    )

    model_trips = GradientBoostingRegressor(random_state=42)
    model_trips.fit(X_train, y_train)

    y_pred_trips = model_trips.predict(X_test)
    y_pred_trips = np.clip(y_pred_trips, 0, None)

    mae_trips = mean_absolute_error(y_test, y_pred_trips)
    logger.info(f"Trips model MAE (time-based split): {mae_trips:.2f}")

    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model_trips, models_dir / "gb_trips.pkl")
    logger.info(f"Saved trips model to {models_dir / 'gb_trips.pkl'}")

    # Feature importance for trips model
    fi_trips = pd.DataFrame(
        {
            "feature": feature_cols_trips,
            "importance": model_trips.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    fi_trips_path = models_dir / "feature_importances_trips.csv"
    fi_trips.to_csv(fi_trips_path, index=False)
    logger.info(f"Saved trips feature importances to {fi_trips_path}")
    logger.info("Top trips features:\n%s", fi_trips.head().to_string(index=False))

    # ----------------------------------------
    # Complaints model (time-based train/test)
    # ----------------------------------------
    feature_cols_complaints = [
        "zone_id",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_friday_night",
        "is_holiday",
        "is_nye",
        "is_halloween",
        "is_payday_window",
        "is_night",
        "avg_trip_distance",
        "avg_total_amount",
        "revenue",
        "trips",
        "lag1_trips",
        "lag1_borough_complaints_total",
    ]

    df_complaints = df.dropna(subset=feature_cols_complaints + ["complaints_total"]).copy()
    df_complaints = df_complaints.sort_values("hour")

    Xc = df_complaints[feature_cols_complaints]
    y_complaints = df_complaints["complaints_total"]

    split_idx2 = int(len(df_complaints) * 0.8)
    Xc_train, Xc_test = Xc.iloc[:split_idx2], Xc.iloc[split_idx2:]
    # FIXED: test labels must align with Xc_test
    y2_train, y2_test = y_complaints.iloc[:split_idx2], y_complaints.iloc[split_idx2:]

    logger.info(
        f"Complaints model training on {len(Xc_train)} rows, "
        f"testing on {len(Xc_test)} rows (time-based split)"
    )

    model_complaints = GradientBoostingRegressor(random_state=42)
    model_complaints.fit(Xc_train, y2_train)

    y2_pred = model_complaints.predict(Xc_test)
    y2_pred = np.clip(y2_pred, 0, None)

    mae_complaints = mean_absolute_error(y2_test, y2_pred)
    logger.info(f"Complaints model MAE (time-based split): {mae_complaints:.2f}")

    joblib.dump(model_complaints, models_dir / "gb_complaints.pkl")
    logger.info(f"Saved complaints model to {models_dir / 'gb_complaints.pkl'}")

    # Feature importance for complaints model
    fi_complaints = pd.DataFrame(
        {
            "feature": feature_cols_complaints,
            "importance": model_complaints.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    fi_complaints_path = models_dir / "feature_importances_complaints.csv"
    fi_complaints.to_csv(fi_complaints_path, index=False)
    logger.info(f"Saved complaints feature importances to {fi_complaints_path}")
    logger.info("Top complaints features:\n%s", fi_complaints.head().to_string(index=False))

    # ---------------------------------------------------
    # Write predictions + scores back to CSV for Tableau
    # ---------------------------------------------------
    df_scores = df.copy()

    # Predict on all rows (fill any NaNs in features)
    X_all_trips = df_scores[feature_cols_trips].fillna(0)
    X_all_complaints = df_scores[feature_cols_complaints].fillna(0)

    df_scores["pred_trips"] = np.clip(
        model_trips.predict(X_all_trips),
        0,
        None,
    )
    df_scores["pred_complaints"] = np.clip(
        model_complaints.predict(X_all_complaints),
        0,
        None,
    )

    # ------------------------
    # Risk / opportunity scores
    # ------------------------

    # Demand score: percentile of predicted trips within (borough, hour_of_day)
    df_scores["demand_score"] = df_scores.groupby(
        ["borough", "hour_of_day"]
    )["pred_trips"].rank(pct=True)

    # Safety score: higher when complaints_per_100_trips is low
    cp = df_scores["complaints_per_100_trips"].copy()

    # Cap very extreme values at 99th percentile to avoid 1–2 crazy points dominating
    cap = cp.quantile(0.99)
    cp_capped = cp.clip(upper=cap)

    global_cp_median = cp_capped.median()
    cp_filled = cp_capped.fillna(global_cp_median)

    df_scores["_cp_for_rank"] = cp_filled

    cp_rank = df_scores.groupby(["borough", "hour_of_day"])["_cp_for_rank"].rank(pct=True)
    df_scores["safety_score"] = 1.0 - cp_rank

    df_scores.drop(columns=["_cp_for_rank"], inplace=True)

    # Risk-adjusted revenue: penalize high complaints
    alpha = 5.0  # penalty weight per complaint
    df_scores["risk_adjusted_revenue"] = (
        df_scores["revenue"]
        - alpha * df_scores["borough_complaints_total"].fillna(0.0)
    )

    # Zone cluster: combine demand & risk
    # High demand = demand_score >= 0.66
    # High risk = complaints_per_100_trips in top ~1/3 (using the capped series)
    risk_threshold = cp_capped.quantile(0.66)
    high_demand = df_scores["demand_score"] >= 0.66
    high_risk = cp_capped.fillna(global_cp_median) >= risk_threshold

    zone_cluster = np.where(
        high_demand & high_risk,
        "High Demand, High Risk",
        np.where(
            high_demand & ~high_risk,
            "High Demand, Low Risk",
            np.where(
                ~high_demand & high_risk,
                "Low Demand, High Risk",
                "Low Demand, Low Risk",
            ),
        ),
    )

    df_scores["zone_cluster"] = zone_cluster

    # Save final wide table for Tableau
    out_csv = GOLD_DIR / "gold_hourly_with_predictions.csv"
    df_scores.to_csv(out_csv, index=False)
    logger.info(f"Wrote predictions + scores to {out_csv}")


if __name__ == "__main__":
    train_and_save_models()
