import os

import mlflow
import mlflow.sklearn
from flask import Flask, jsonify, request


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:////workspaces/MLOPS/mlflow.db",
)

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/taxi-duration-regressor@champion",
)


mlflow.set_tracking_uri(TRACKING_URI)

print(f"Loading model from: {MODEL_URI}")

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception as error:
    raise RuntimeError(
        f"Could not load model from {MODEL_URI}. "
        f"Tracking URI: {TRACKING_URI}"
    ) from error


app = Flask(__name__)


def prepare_ride(ride):
    return {
        "PULocationID": str(ride["PULocationID"]),
        "DOLocationID": str(ride["DOLocationID"]),
        "trip_distance": float(ride["trip_distance"]),
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": True,
            "model_uri": MODEL_URI,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    ride = request.get_json()

    if ride is None:
        return jsonify(
            {
                "error": "Request body must contain JSON."
            }
        ), 400

    required_fields = [
        "PULocationID",
        "DOLocationID",
        "trip_distance",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in ride
    ]

    if missing_fields:
        return jsonify(
            {
                "error": "Missing required fields.",
                "missing_fields": missing_fields,
            }
        ), 400

    try:
        prepared_ride = prepare_ride(ride)

        prediction = model.predict(
            [prepared_ride]
        )[0]

    except (TypeError, ValueError) as error:
        return jsonify(
            {
                "error": "Invalid input values.",
                "details": str(error),
            }
        ), 400

    return jsonify(
        {
            "predicted_duration": float(prediction),
            "unit": "minutes",
            "model_uri": MODEL_URI,
        }
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=9696,
        debug=False,
    )