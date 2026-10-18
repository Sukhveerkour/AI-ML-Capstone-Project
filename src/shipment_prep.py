
import numpy as np
import pandas as pd

PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96,
}

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


def clean_shipments(raw, medians=None):
    """Clean shipment data and return the cleaned data and imputation medians."""
    df = raw.copy()

    # Remove exact duplicate rows.
    df = df.drop_duplicates().reset_index(drop=True)

    # Standardize service names.
    df["service_type"] = df["service_type"].astype("string").str.strip().str.title()

    # Fill missing payment modes.
    df["payment_mode"] = df["payment_mode"].fillna("Unknown").astype(str).str.strip()

    # Convert rupee-formatted declared values into numeric values.
    df["declared_value_inr"] = pd.to_numeric(
        df["declared_value"]
        .astype("string")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    ).astype(float)
    df = df.drop(columns=["declared_value"])

    # Parse dispatch timestamps.
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], errors="coerce")

    # Treat impossible sentinel weights as missing.
    df["parcel_weight_kg"] = pd.to_numeric(
        df["parcel_weight_kg"], errors="coerce"
    )
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")

    # Calculate medians from historical data, or reuse supplied medians.
    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median()),
            "parcel_weight_kg": float(df["parcel_weight_kg"].median()),
            "declared_value_inr": float(df["declared_value_inr"].median()),
        }
    else:
        medians = {key: float(value) for key, value in medians.items()}

    # Fill missing numerical values.
    for column in ["distance_km", "parcel_weight_kg", "declared_value_inr"]:
        df[column] = df[column].fillna(medians[column])

    return df, medians


def build_features(clean, hubs):
    """Add hub information and model features without using delivery_hours."""
    df = clean.copy()

    hub_columns = [
        "hub_id",
        "hub_city",
        "region",
        "hub_tier",
        "hub_daily_capacity",
    ]

    df = df.merge(
        hubs[hub_columns],
        on="hub_id",
        how="left",
        validate="many_to_one",
    )

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df
