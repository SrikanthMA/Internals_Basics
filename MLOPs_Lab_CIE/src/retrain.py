import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

EXPERIMENT_NAME = "chemreact-reaction-yield-pct"
FEATURES = ["temperature_c", "pressure_atm", "catalyst_type_index", "flow_rate_lph"]
TARGET = "reaction_yield_pct"

# Load step1 to know which model won
with open(os.path.join(RESULTS_DIR, "step1_s1.json")) as f:
    step1 = json.load(f)

best_model_name = step1["best_model"]
champion_mae = None
for m in step1["models"]:
    if m["name"] == best_model_name:
        champion_mae = m["mae"]

# Load data
train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")
combined_df = pd.concat([train_df, new_df], ignore_index=True)

original_rows = len(train_df)
new_rows = len(new_df)
combined_rows = len(combined_df)

# Use SAME test split as original (on combined data, same random_state)
X = combined_df[FEATURES]
y = combined_df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the winning model type
if best_model_name == "Ridge":
    retrain_model = Ridge(alpha=1.0)
else:
    retrain_model = LinearRegression()

mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name=f"{best_model_name}_retrained") as run:
    mlflow.set_tag("priority", "high")
    retrain_model.fit(X_train, y_train)
    preds = retrain_model.predict(X_test)

    retrained_mae = mean_absolute_error(y_test, preds)
    retrained_rmse = np.sqrt(mean_squared_error(y_test, preds))

    mlflow.log_metric("mae", retrained_mae)
    mlflow.log_metric("rmse", retrained_rmse)
    mlflow.sklearn.log_model(retrain_model, artifact_path="model")

improvement = champion_mae - retrained_mae
min_threshold = 0.5
action = "promoted" if improvement >= min_threshold else "kept_champion"

if action == "promoted":
    joblib.dump(retrain_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    print("Retrained model promoted as new champion.")
else:
    print("Champion kept — improvement insufficient.")

output = {
    "original_data_rows": original_rows,
    "new_data_rows": new_rows,
    "combined_data_rows": combined_rows,
    "champion_mae": round(champion_mae, 4),
    "retrained_mae": round(retrained_mae, 4),
    "improvement": round(improvement, 4),
    "min_improvement_threshold": min_threshold,
    "action": action,
    "comparison_metric": "mae",
}
with open(os.path.join(RESULTS_DIR, "step4_s8.json"), "w") as f:
    json.dump(output, f, indent=2)

print("Saved results/step4_s8.json")