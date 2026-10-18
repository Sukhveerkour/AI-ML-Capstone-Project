
import os
import sys
import numpy as np
import pandas as pd

# Allow imports from the project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.shipment_prep import clean_shipments, build_features

DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the original generated data
hubs = pd.read_csv(os.path.join(DATA_DIR, "hubs.csv"))
raw = pd.read_csv(os.path.join(DATA_DIR, "raw_shipments.csv"))

# Clean data and engineer features
clean, medians = clean_shipments(raw)
featured = build_features(clean, hubs)

# Determine whether each shipment was delayed
featured["is_delayed"] = (
    featured["delivery_hours"] > featured["promised_hours"]
).astype(int)

# Save the modelling table
featured.to_csv(os.path.join(DATA_DIR, "modelling_table.csv"), index=False)

# Data quality summary
quality = pd.DataFrame([
    {"check": "raw_rows", "value": len(raw)},
    {"check": "duplicate_rows_removed", "value": int(raw.duplicated().sum())},
    {"check": "cleaned_rows", "value": len(clean)},
    {"check": "featured_rows", "value": len(featured)},
    {"check": "missing_distance_before_cleaning",
     "value": int(raw["distance_km"].isna().sum())},
    {"check": "missing_weight_before_cleaning",
     "value": int(raw["parcel_weight_kg"].isna().sum())},
    {"check": "missing_declared_value_before_cleaning",
     "value": int(raw["declared_value"].isna().sum())},
    {"check": "missing_payment_mode_before_cleaning",
     "value": int(raw["payment_mode"].isna().sum())},
    {"check": "sentinel_weight_rows",
     "value": int((pd.to_numeric(raw["parcel_weight_kg"], errors="coerce") > 100).sum())},
    {"check": "delayed_count", "value": int(featured["is_delayed"].sum())},
    {"check": "delay_rate", "value": round(float(featured["is_delayed"].mean()), 4)},
])

quality.to_csv(
    os.path.join(OUTPUT_DIR, "part_a_data_quality.csv"), index=False
)

# Create distance bands for analysis
featured["distance_band"] = pd.cut(
    featured["distance_km"],
    bins=[0, 200, 400, 600, np.inf],
    labels=["0-200", "201-400", "401-600", "601+"],
    include_lowest=True,
)

# Summarize delay patterns across important groups
group_specs = [
    ("service_type", "service_type"),
    ("payment_mode", "payment_mode"),
    ("hub_tier", "hub_tier"),
    ("rain_flag", "rain_flag"),
    ("sale_event_flag", "sale_event_flag"),
    ("distance_band", "distance_band"),
]

reports = []

for group_name, column in group_specs:
    grouped = featured.groupby(column, observed=True).agg(
        count=("shipment_id", "size"),
        delayed_count=("is_delayed", "sum"),
        mean_delivery_hours=("delivery_hours", "mean"),
    ).reset_index()

    grouped = grouped.rename(columns={column: "group_value"})
    grouped.insert(0, "group_type", group_name)
    grouped["delay_rate"] = grouped["delayed_count"] / grouped["count"]
    reports.append(grouped)

delay_report = pd.concat(reports, ignore_index=True)
delay_report.to_csv(
    os.path.join(OUTPUT_DIR, "part_a_delay_by_group.csv"), index=False
)

# Correlation checks for selected numerical columns
correlation_pairs = [
    ("delivery_hours", "promised_hours"),
    ("delivery_hours", "distance_km"),
    ("is_delayed", "distance_km"),
]

correlation_rows = []
for first, second in correlation_pairs:
    correlation_rows.append({
        "variable_1": first,
        "variable_2": second,
        "correlation": featured[first].corr(featured[second]),
    })

correlations = pd.DataFrame(correlation_rows)
correlations.to_csv(
    os.path.join(OUTPUT_DIR, "part_a_correlations.csv"), index=False
)

# Print a summary for verification
print("PART A COMPLETED")
print("Raw rows:", len(raw))
print("Cleaned rows:", len(clean))
print("Modelling table shape:", featured.shape)
print("Medians:", medians)
print("Delayed shipments:", int(featured["is_delayed"].sum()))
print("Delay rate:", round(float(featured["is_delayed"].mean()), 4))
print("\nData quality report:")
print(quality.to_string(index=False))
print("\nDelay by group:")
print(delay_report.to_string(index=False))
print("\nCorrelation checks:")
print(correlations.to_string(index=False))
print("\nFiles saved:")
print("data/modelling_table.csv")
print("outputs/part_a_data_quality.csv")
print("outputs/part_a_delay_by_group.csv")
print("outputs/part_a_correlations.csv")
