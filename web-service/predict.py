import pickle
from pathlib import Path

from flask import Flask, jsonify, request


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "taxi_duration_model.bin"


with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


app = Flask(__name__)


def prepare_ride(ride):
    return {
        "PULocationID": str(ride["PULocationID"]),
        "DOLocationID": str(ride["DOLocationID"]),
        "trip_distance": float(ride["trip_distance"]),
    }


@app.route("/predict", methods=["POST"])
def predict():
    ride = request.get_json()

    if ride is None:
        return jsonify(
            {"error": "Request body must contain JSON."}
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

    prepared_ride = prepare_ride(ride)

    predicted_duration = model.predict(
        [prepared_ride]
    )[0]

    result = {
        "predicted_duration": float(predicted_duration),
        "unit": "minutes",
    }

    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": True,
        }
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=9696,
        debug=True,
    )
    