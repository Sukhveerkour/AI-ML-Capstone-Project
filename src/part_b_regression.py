
import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.shipment_prep import FEATURES

DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "modelling_table.csv"))

X = df[FEATURES].copy()
y = df["delivery_hours"].copy()

# Split into 70% training and 30% temporary data.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Split temporary data equally into validation and test sets.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Split sizes:", len(X_train), len(X_val), len(X_test))

def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }

# Mean baseline
baseline_pred = np.full(len(y_val), y_train.mean())
baseline_metrics = metrics(y_val, baseline_pred)

results = [{
    "model": "Mean baseline",
    "validation_MAE": baseline_metrics["MAE"],
    "validation_RMSE": baseline_metrics["RMSE"],
    "validation_R2": baseline_metrics["R2"],
    "tree_depth": np.nan,
}]

# Select the decision tree depth using validation RMSE.
depth_results = []
for depth in [1, 2, 3, 4, 6, 8, 12, None]:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    pred = tree.predict(X_val)
    m = metrics(y_val, pred)
    depth_results.append({
        "max_depth": "unlimited" if depth is None else depth,
        "validation_MAE": m["MAE"],
        "validation_RMSE": m["RMSE"],
        "validation_R2": m["R2"],
    })

depth_df = pd.DataFrame(depth_results)
depth_df.to_csv(
    os.path.join(OUTPUT_DIR, "part_b_tree_depths.csv"), index=False
)

best_depth_row = depth_df.sort_values(
    ["validation_RMSE", "max_depth"],
    key=lambda col: pd.to_numeric(col, errors="coerce").fillna(999)
    if col.name == "max_depth" else col,
).iloc[0]

best_depth_value = best_depth_row["max_depth"]
best_depth = None if best_depth_value == "unlimited" else int(best_depth_value)

# Candidate regression models
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=best_depth, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
}

fitted_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model
    pred = model.predict(X_val)
    m = metrics(y_val, pred)
    results.append({
        "model": name,
        "validation_MAE": m["MAE"],
        "validation_RMSE": m["RMSE"],
        "validation_R2": m["R2"],
        "tree_depth": best_depth if name == "Decision Tree" else np.nan,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(OUTPUT_DIR, "part_b_validation_metrics.csv"), index=False
)

# Learning-rate experiment for Gradient Boosting
lr_results = []
for rate in [0.01, 0.1, 1.0]:
    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=rate, random_state=42
    )
    model.fit(X_train, y_train)
    m = metrics(y_val, model.predict(X_val))
    lr_results.append({
        "learning_rate": rate,
        "validation_MAE": m["MAE"],
        "validation_RMSE": m["RMSE"],
        "validation_R2": m["R2"],
    })

pd.DataFrame(lr_results).to_csv(
    os.path.join(OUTPUT_DIR, "part_b_learning_rates.csv"), index=False
)

# Select the winner using validation RMSE; do not use the test set for selection.
candidate_results = results_df[
    results_df["model"] != "Mean baseline"
].sort_values("validation_RMSE")
winner_name = candidate_results.iloc[0]["model"]
winner = fitted_models[winner_name]

# Cross-validation of the two strongest candidates on training + validation data.
X_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = pd.concat([y_train, y_val], axis=0)

top_two = candidate_results["model"].head(2).tolist()
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name in top_two:
    model = models[name]
    scores = cross_val_score(
        model, X_trainval, y_trainval,
        cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    cv_results.append({
        "model": name,
        "mean_cv_RMSE": -scores.mean(),
        "std_cv_RMSE": scores.std(),
    })

pd.DataFrame(cv_results).to_csv(
    os.path.join(OUTPUT_DIR, "part_b_cross_validation.csv"), index=False
)

# Evaluate the chosen model on the test set once.
test_metrics = metrics(y_test, winner.predict(X_test))
pd.DataFrame([{
    "winner": winner_name,
    "test_MAE": test_metrics["MAE"],
    "test_RMSE": test_metrics["RMSE"],
    "test_R2": test_metrics["R2"],
}]).to_csv(
    os.path.join(OUTPUT_DIR, "part_b_final_test_metrics.csv"), index=False
)

print("\nValidation metrics:")
print(results_df.round(4).to_string(index=False))
print("\nDecision tree depth comparison:")
print(depth_df.round(4).to_string(index=False))
print("\nGradient Boosting learning-rate comparison:")
print(pd.DataFrame(lr_results).round(4).to_string(index=False))
print("\nCross-validation:")
print(pd.DataFrame(cv_results).round(4).to_string(index=False))
print("\nSelected model:", winner_name)
print("Final test metrics:", test_metrics)
print("\nPart B reports saved in outputs/")
