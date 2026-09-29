import argparse
import uuid
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd


TRACKING_URI = "sqlite:////workspaces/MLOPS/mlflow.db"

DEFAULT_MODEL_URI = (
    "models:/taxi-duration-regressor@champion"
)

CATEGORICAL_FEATURES = [
    "PULocationID",
    "DOLocationID",
]

NUMERICAL_FEATURES = [
    "trip_distance",
]


def read_data(input_file, limit):
    """Read and prepare taxi rides for batch scoring."""

    print(f"Reading data from: {input_file}")

    required_columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "trip_distance",
    ]

    df = pd.read_parquet(
        input_file,
        columns=required_columns,
    )

    print(f"Rows loaded: {len(df):,}")

    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"]
    )

    df["tpep_dropoff_datetime"] = pd.to_datetime(
        df["tpep_dropoff_datetime"]
    )

    df["actual_duration"] = (
        df["tpep_dropoff_datetime"]
        - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    valid_duration = (
        (df["actual_duration"] >= 1)
        & (df["actual_duration"] <= 60)
    )

    df = df.loc[valid_duration].copy()

    print(f"Rows after filtering: {len(df):,}")

    if limit > 0 and len(df) > limit:
        df = df.sample(
            n=limit,
            random_state=42,
        ).copy()

        print(f"Rows used after sampling: {len(df):,}")

    df[CATEGORICAL_FEATURES] = (
        df[CATEGORICAL_FEATURES]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    df[NUMERICAL_FEATURES] = (
        df[NUMERICAL_FEATURES]
        .fillna(0)
    )

    df["ride_id"] = [
        str(uuid.uuid4())
        for _ in range(len(df))
    ]

    return df


def prepare_features(df):
    """Create dictionaries expected by the saved model."""

    feature_columns = (
        CATEGORICAL_FEATURES
        + NUMERICAL_FEATURES
    )

    return df[
        feature_columns
    ].to_dict(orient="records")


def load_model(model_uri):
    """Load the selected model from MLflow."""

    mlflow.set_tracking_uri(TRACKING_URI)

    print(f"Loading model from: {model_uri}")

    model = mlflow.sklearn.load_model(
        model_uri
    )

    return model


def save_results(
    df,
    predictions,
    model_uri,
    output_file,
):
    """Save the batch predictions to a Parquet file."""

    result = pd.DataFrame()

    result["ride_id"] = df["ride_id"].values

    result["pickup_datetime"] = (
        df["tpep_pickup_datetime"].values
    )

    result["PULocationID"] = (
        df["PULocationID"].values
    )

    result["DOLocationID"] = (
        df["DOLocationID"].values
    )

    result["trip_distance"] = (
        df["trip_distance"].values
    )

    result["actual_duration"] = (
        df["actual_duration"].values
    )

    result["predicted_duration"] = predictions

    result["prediction_error"] = (
        result["actual_duration"]
        - result["predicted_duration"]
    )

    result["model_uri"] = model_uri

    output_path = Path(output_file)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        output_path,
        index=False,
    )

    print(f"Predictions saved to: {output_path}")
    print(f"Rows written: {len(result):,}")
    print("\nFirst five predictions:")
    print(result.head())

    return result


def run(
    input_file,
    output_file,
    model_uri,
    limit,
):
    """Run the complete batch-scoring process."""

    df = read_data(
        input_file=input_file,
        limit=limit,
    )

    feature_dicts = prepare_features(df)

    model = load_model(model_uri)

    print("Calculating batch predictions...")

    predictions = model.predict(
        feature_dicts
    )

    result = save_results(
        df=df,
        predictions=predictions,
        model_uri=model_uri,
        output_file=output_file,
    )

    print(
        "\nAverage predicted duration: "
        f"{result['predicted_duration'].mean():.2f} minutes"
    )

    print(
        "Average actual duration: "
        f"{result['actual_duration'].mean():.2f} minutes"
    )

    print("\nBatch scoring completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate batch taxi-duration predictions."
        )
    )

    parser.add_argument(
        "--input-file",
        required=True,
        help="Input Parquet file.",
    )

    parser.add_argument(
        "--output-file",
        required=True,
        help="Output predictions Parquet file.",
    )

    parser.add_argument(
        "--model-uri",
        default=DEFAULT_MODEL_URI,
        help="MLflow model URI.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=75_000,
        help=(
            "Maximum rows to score. "
            "Use 0 to process all rows."
        ),
    )

    args = parser.parse_args()

    run(
        input_file=args.input_file,
        output_file=args.output_file,
        model_uri=args.model_uri,
        limit=args.limit,
    )