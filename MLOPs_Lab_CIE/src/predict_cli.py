"""
Task 2: CLI predictor — runs inside Docker.
Usage:
  python predict_cli.py --temperature_c 133.0 --pressure_atm 5.3 \
                        --catalyst_type_index 3 --flow_rate_lph 78.3
"""

import argparse
import json
import os
import pickle

# model lives at /app/models/best_model.pkl inside Docker
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "best_model.pkl")

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description="ChemReact yield predictor")
    parser.add_argument("--temperature_c",      type=float, required=True)
    parser.add_argument("--pressure_atm",       type=float, required=True)
    parser.add_argument("--catalyst_type_index",type=float, required=True)
    parser.add_argument("--flow_rate_lph",      type=float, required=True)
    args = parser.parse_args()

    model = load_model()
    features = [[args.temperature_c, args.pressure_atm,
                 args.catalyst_type_index, args.flow_rate_lph]]
    prediction = float(model.predict(features)[0])

    result = {
        "temperature_c":       args.temperature_c,
        "pressure_atm":        args.pressure_atm,
        "catalyst_type_index": args.catalyst_type_index,
        "flow_rate_lph":       args.flow_rate_lph,
        "predicted_reaction_yield_pct": round(prediction, 4),
    }
    print(json.dumps(result, indent=2))
    return prediction

if __name__ == "__main__":
    main()
