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

# Paths
DATA_PATH = "data/training_data.csv"
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
X = df[["temperature_c", "pressure_atm", "catalyst_type_index", "flow_rate_lph"]]
y = df["reaction_yield_pct"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

EXPERIMENT_NAME = "chemreact-reaction-yield-pct"
mlflow.set_experiment(EXPERIMENT_NAME)

models_config = [
    ("LinearRegression", LinearRegression(), {}),
    ("Ridge", Ridge(alpha=1.0), {"alpha": 1.0}),
]

results = []
best_model_name = None
best_rmse = float("inf")
best_mae = None
best_run_id = None

for model_name, model, params in models_config:
    with mlflow.start_run(run_name=model_name) as run:
        # Log params
        mlflow.log_params(params if params else {"fit_intercept": True})
        mlflow.set_tag("priority", "high")

        # Train
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, artifact_path="model")

        results.append({"name": model_name, "mae": round(mae, 4), "rmse": round(rmse, 4), "run_id": run.info.run_id})

        if rmse < best_rmse:
            best_rmse = rmse
            best_mae = mae
            best_model_name = model_name
            best_run_id = run.info.run_id
            best_model_obj = model

# Save best model
joblib.dump(best_model_obj, os.path.join(MODELS_DIR, "best_model.pkl"))
print(f"Best model: {best_model_name} (RMSE={round(best_rmse,4)})")

# Save results JSON
output = {
    "experiment_name": EXPERIMENT_NAME,
    "models": [{"name": r["name"], "mae": r["mae"], "rmse": r["rmse"]} for r in results],
    "best_model": best_model_name,
    "best_metric_name": "rmse",
    "best_metric_value": round(best_rmse, 4),
}
with open(os.path.join(RESULTS_DIR, "step1_s1.json"), "w") as f:
    json.dump(output, f, indent=2)

print("Saved results/step1_s1.json")