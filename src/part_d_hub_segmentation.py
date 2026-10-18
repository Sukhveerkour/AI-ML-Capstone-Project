import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

ROOT = "/content/AI-ML-Capstone-Project"
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "modelling_table.csv"))

# Hub-level profile
profile = (
    df.groupby("hub_id")
    .agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_stops=("n_stops", "mean"),
    )
    .reset_index()
    .sort_values("hub_id")
    .reset_index(drop=True)
)

features = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_stops",
]

scaler = StandardScaler()
X = scaler.fit_transform(profile[features])

# Three hub clusters
model = KMeans(n_clusters=3, random_state=42, n_init=10)
profile["cluster"] = model.fit_predict(X)

# Label clusters from lowest to highest delay rate
order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
names = {
    order[0]: "Standard",
    order[1]: "Watch",
    order[2]: "Escalate",
}
profile["hub_segment"] = profile["cluster"].map(names)

profile.to_csv(
    os.path.join(OUTPUT_DIR, "part_d_hub_segments.csv"),
    index=False,
)

summary = (
    profile.groupby("hub_segment")
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_stops=("mean_stops", "mean"),
    )
    .reindex(["Standard", "Watch", "Escalate"])
    .reset_index()
)

summary.to_csv(
    os.path.join(OUTPUT_DIR, "part_d_segment_summary.csv"),
    index=False,
)

# Compare separation ratios for K = 2, 3, 4, 5
def separation_ratio(X, labels):
    distances = pairwise_distances(X)
    within, between = [], []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                within.append(distances[i, j])
            else:
                between.append(distances[i, j])

    if not within or np.mean(within) == 0:
        return np.nan

    return np.mean(between) / np.mean(within)

ratios = []
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    ratios.append({
        "k": k,
        "separation_ratio": separation_ratio(X, labels),
        "cluster_sizes": "-".join(
            map(str, sorted(pd.Series(labels).value_counts(), reverse=True))
        ),
    })

ratio_df = pd.DataFrame(ratios)
ratio_df.to_csv(
    os.path.join(OUTPUT_DIR, "part_d_separation_ratios.csv"),
    index=False,
)

print("PART D HUB SEGMENTATION COMPLETED")
print("\nNumber of hubs:", len(profile))
print("\nHub segments:")
print(profile[["hub_id", "cluster", "hub_segment"]].to_string(index=False))
print("\nSegment summary:")
print(summary.round(4).to_string(index=False))
print("\nSeparation ratios:")
print(ratio_df.round(4).to_string(index=False))
print("\nReports saved in outputs/")
