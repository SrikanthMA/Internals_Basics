import json, os, joblib, numpy as np

model = joblib.load("models/best_model.pkl")
pred = model.predict([[133.0, 5.3, 3, 78.3]])[0]

output = {
    "image_name": "chemreact-predictor",
    "image_tag": "v1",
    "base_image": "python:3.10-slim",
    "test_input": {
        "temperature_c": 133.0,
        "pressure_atm": 5.3,
        "catalyst_type_index": 3,
        "flow_rate_lph": 78.3,
    },
    "prediction": round(float(pred), 4),
}
os.makedirs("results", exist_ok=True)
with open("results/step2_s3.json", "w") as f:
    json.dump(output, f, indent=2)
print("Saved results/step2_s3.json")