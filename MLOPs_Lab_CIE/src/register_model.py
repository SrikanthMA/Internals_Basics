import os
import json
import mlflow
from mlflow.tracking import MlflowClient

RESULTS_DIR = "results"
EXPERIMENT_NAME = "chemreact-reaction-yield-pct"
REGISTERED_MODEL_NAME = "chemreact-reaction-yield-pct-predictor"

# Load step1 results to get best model info
with open(os.path.join(RESULTS_DIR, "step1_s1.json")) as f:
    step1 = json.load(f)

best_model_name = step1["best_model"]
best_rmse = step1["best_metric_value"]

# Find the run_id of the best model
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{best_model_name}'",
    order_by=["metrics.rmse ASC"],
    max_results=1,
)

best_run = runs[0]
run_id = best_run.info.run_id
model_uri = f"runs:/{run_id}/model"

# Register model
model_version = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
version_number = int(model_version.version)

print(f"Registered model: {REGISTERED_MODEL_NAME}, version: {version_number}, run_id: {run_id}")

# Save results
output = {
    "registered_model_name": REGISTERED_MODEL_NAME,
    "version": version_number,
    "run_id": run_id,
    "source_metric": "rmse",
    "source_metric_value": best_rmse,
}
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, "step3_s6.json"), "w") as f:
    json.dump(output, f, indent=2)

print("Saved results/step3_s6.json")