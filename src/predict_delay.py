import os
import sys
import json
import joblib
import pandas as pd

ROOT = "/content/AI-ML-Capstone-Project"
sys.path.insert(0, ROOT)

from src.shipment_prep import clean_shipments, build_features

DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(ROOT, "models")

# Load the original data and new shipments
raw = pd.read_csv(os.path.join(DATA_DIR, "raw_shipments.csv"))
new_shipments = pd.read_csv(os.path.join(DATA_DIR, "new_shipments.csv"))
hubs = pd.read_csv(os.path.join(DATA_DIR, "hubs.csv"))

# Reuse the training-data cleaning medians
_, medians = clean_shipments(raw)

# Clean and create features for new shipments
clean_new, _ = clean_shipments(new_shipments, medians=medians)
featured_new = build_features(clean_new, hubs)

# Load the packaged model, feature order and threshold
model = joblib.load(os.path.join(MODEL_DIR, "delay_classifier.joblib"))

feature_order = pd.read_csv(
    os.path.join(MODEL_DIR, "feature_order.csv")
)["feature"].tolist()

with open(os.path.join(MODEL_DIR, "decision_threshold.json"), "r") as f:
    threshold_config = json.load(f)

threshold = float(threshold_config["threshold"])

# Predict delay probabilities
X_new = featured_new[feature_order]
probabilities = model.predict_proba(X_new)[:, 1]
predictions = (probabilities >= threshold).astype(int)

# Add hub operational segments
segments = pd.read_csv(
    os.path.join(OUTPUT_DIR, "part_d_hub_segments.csv")
)[["hub_id", "hub_segment"]]

results = featured_new[["shipment_id", "hub_id"]].copy()
results["delay_probability"] = probabilities.round(4)
results["predicted_delayed"] = predictions

results = results.merge(segments, on="hub_id", how="left")

# Recommend an operational action
def recommend_action(row):
    if row["predicted_delayed"] == 1:
        if row["hub_segment"] == "Escalate":
            return "Escalate to hub manager"
        return "Proactive customer call"
    return "No action"

results["recommended_action"] = results.apply(recommend_action, axis=1)

output_path = os.path.join(OUTPUT_DIR, "new_shipment_actions.csv")
results.to_csv(output_path, index=False)

print("NEW SHIPMENT PREDICTIONS COMPLETED")
print("New shipment rows:", len(results))
print("Missing values:", int(results.isna().sum().sum()))
print("Predicted delayed:", int(results["predicted_delayed"].sum()))
print("\nRecommended actions:")
print(results["recommended_action"].value_counts().to_string())
print("\nSample predictions:")
print(results.head(10).to_string(index=False))
print("\nSaved:", output_path)
