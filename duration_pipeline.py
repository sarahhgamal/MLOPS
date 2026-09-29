import argparse
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import pickle


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "data").exists():
    DATA_DIR = SCRIPT_DIR / "data"
else:
    DATA_DIR = SCRIPT_DIR

MLFLOW_DATABASE = SCRIPT_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DATABASE}"

CATEGORICAL_FEATURES = [
    "PULocationID",
    "DOLocationID",
]

NUMERICAL_FEATURES = [
    "trip_distance",
]

MIN_DURATION = 1
MAX_DURATION = 60
SAMPLE_SIZE = 75_000
RANDOM_STATE = 42


# --------------------------------------------------
# Date and file-path preparation
# --------------------------------------------------

def get_next_month(year, month):
    """Return the year and month following the supplied period."""

    if month == 12:
        return year + 1, 1

    return year, month + 1


def get_data_paths(year, month):
    """
    Build the training and validation file paths.

    The selected month is used for training.
    The following month is used for validation.
    """

    validation_year, validation_month = get_next_month(
        year,
        month,
    )

    training_file = DATA_DIR / (
        f"yellow_tripdata_{year}-{month:02d}.parquet"
    )

    validation_file = DATA_DIR / (
        f"yellow_tripdata_"
        f"{validation_year}-{validation_month:02d}.parquet"
    )

    if not training_file.exists():
        raise FileNotFoundError(
            f"Training file not found:\n{training_file}\n\n"
            "Download the required Parquet file and place it "
            f"inside:\n{DATA_DIR}"
        )

    if not validation_file.exists():
        raise FileNotFoundError(
            f"Validation file not found:\n{validation_file}\n\n"
            "The pipeline trains on the selected month and "
            "validates on the following month."
        )

    return (
        training_file,
        validation_file,
        validation_year,
        validation_month,
    )


# --------------------------------------------------
# Data preparation
# --------------------------------------------------
def prepare_data(file_path):
    """Load and prepare one yellow taxi dataset."""

    print(f"\nLoading {file_path.name}...")

    required_columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "trip_distance",
    ]

    # Load only the five columns required by the model.
    # This significantly reduces memory usage.
    df = pd.read_parquet(
        file_path,
        columns=required_columns,
    )

    print(f"Original number of rows: {len(df):,}")

    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"]
    )

    df["tpep_dropoff_datetime"] = pd.to_datetime(
        df["tpep_dropoff_datetime"]
    )

    # Calculate duration in minutes.
    df["duration"] = (
        df["tpep_dropoff_datetime"]
        - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # Keep trips between 1 and 60 minutes.
    duration_mask = (
        (df["duration"] >= MIN_DURATION)
        & (df["duration"] <= MAX_DURATION)
    )

    df = df.loc[
        duration_mask,
        CATEGORICAL_FEATURES
        + NUMERICAL_FEATURES
        + ["duration"],
    ].copy()

    print(f"Rows after filtering: {len(df):,}")

    # Treat location IDs as categories.
    df[CATEGORICAL_FEATURES] = (
        df[CATEGORICAL_FEATURES]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    # Replace missing distance values.
    df[NUMERICAL_FEATURES] = (
        df[NUMERICAL_FEATURES]
        .fillna(0)
    )

    rows_to_sample = min(SAMPLE_SIZE, len(df))

    df = df.sample(
        n=rows_to_sample,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print(f"Rows used after sampling: {len(df):,}")
    print(
        f"Average duration: "
        f"{df['duration'].mean():.2f} minutes"
    )

    return df

# --------------------------------------------------
# Feature preparation
# --------------------------------------------------

def prepare_features(df_train, df_val):
    """Create the feature dictionaries and target arrays."""

    selected_features = (
        CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    )

    train_dicts = df_train[
        selected_features
    ].to_dict(orient="records")

    val_dicts = df_val[
        selected_features
    ].to_dict(orient="records")

    y_train = df_train["duration"].to_numpy()
    y_val = df_val["duration"].to_numpy()

    print("\nFeature preparation completed.")
    print(f"Training rows: {len(train_dicts):,}")
    print(f"Validation rows: {len(val_dicts):,}")

    return train_dicts, val_dicts, y_train, y_val


# --------------------------------------------------
# Model training
# --------------------------------------------------

def train_model(train_dicts, y_train):
    """
    Train a pipeline containing both the DictVectorizer
    and LinearRegression model.
    """

    print("\nTraining Linear Regression...")

    model_pipeline = make_pipeline(
        DictVectorizer(),
        LinearRegression(),
    )

    model_pipeline.fit(
        train_dicts,
        y_train,
    )

    return model_pipeline


# --------------------------------------------------
# Model evaluation
# --------------------------------------------------

def calculate_rmse(model, feature_dicts, target):
    """Generate predictions and calculate RMSE."""

    predictions = model.predict(feature_dicts)

    rmse = np.sqrt(
        mean_squared_error(
            target,
            predictions,
        )
    )

    return rmse


# --------------------------------------------------
# Complete pipeline
# --------------------------------------------------

def run(year, month):
    """Run the complete parameterized training pipeline."""

    if month < 1 or month > 12:
        raise ValueError("Month must be between 1 and 12.")

    (
        training_file,
        validation_file,
        validation_year,
        validation_month,
    ) = get_data_paths(
        year,
        month,
    )

    print("\nNYC Taxi Duration Training Pipeline")
    print("-----------------------------------")
    print(f"Training period: {year}-{month:02d}")
    print(
        "Validation period: "
        f"{validation_year}-{validation_month:02d}"
    )
    print(f"Training file: {training_file}")
    print(f"Validation file: {validation_file}")

    # Step 1: Load and prepare data.
    df_train = prepare_data(training_file)
    df_val = prepare_data(validation_file)

    # Step 2: Prepare model inputs and targets.
    (
        train_dicts,
        val_dicts,
        y_train,
        y_val,
    ) = prepare_features(
        df_train,
        df_val,
    )

    # Step 3: Train the complete preprocessing/model pipeline.
    model_pipeline = train_model(
        train_dicts,
        y_train,
    )

    # Step 4: Evaluate the model.
    train_rmse = calculate_rmse(
        model_pipeline,
        train_dicts,
        y_train,
    )

    val_rmse = calculate_rmse(
        model_pipeline,
        val_dicts,
        y_val,
    )

    print("\nLinear Regression results:")
    print(
        f"Training RMSE: {train_rmse:.2f} minutes"
    )
    print(
        f"Validation RMSE: {val_rmse:.2f} minutes"
    )

    # Step 5: Track the execution in MLflow.
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        "nyc-taxi-pipeline"
    )

    run_name = (
        f"linear-regression-"
        f"{year}-{month:02d}"
    )

    with mlflow.start_run(run_name=run_name):

        # Data-period parameters.
        mlflow.log_param("training_year", year)
        mlflow.log_param("training_month", month)
        mlflow.log_param(
            "validation_year",
            validation_year,
        )
        mlflow.log_param(
            "validation_month",
            validation_month,
        )

        # Exact data references.
        mlflow.log_param(
            "train_data",
            str(training_file),
        )
        mlflow.log_param(
            "valid_data",
            str(validation_file),
        )

        # Model parameters.
        mlflow.log_param(
            "model",
            "LinearRegression",
        )

        # Preprocessing parameters.
        mlflow.log_param(
            "minimum_duration",
            MIN_DURATION,
        )
        mlflow.log_param(
            "maximum_duration",
            MAX_DURATION,
        )
        mlflow.log_param(
            "sample_size",
            SAMPLE_SIZE,
        )
        mlflow.log_param(
            "sample_random_state",
            RANDOM_STATE,
        )

        mlflow.log_param(
            "categorical_features",
            ",".join(CATEGORICAL_FEATURES),
        )

        mlflow.log_param(
            "numerical_features",
            ",".join(NUMERICAL_FEATURES),
        )

        mlflow.log_param(
            "training_rows",
            len(df_train),
        )

        mlflow.log_param(
            "validation_rows",
            len(df_val),
        )

        # Evaluation metrics.
        mlflow.log_metric(
            "training_rmse",
            train_rmse,
        )

        mlflow.log_metric(
            "validation_rmse",
            val_rmse,
        )

        # Log the complete pipeline.
        # This includes both DictVectorizer and LinearRegression.
        mlflow.sklearn.log_model(
            model_pipeline,
            name="taxi-duration-model",
        )

        active_run = mlflow.active_run()

        print("\nPipeline run saved to MLflow.")
        print(f"MLflow run ID: {active_run.info.run_id}")

        models_dir = SCRIPT_DIR / "models"
        models_dir.mkdir(exist_ok=True)

        model_file = models_dir / "taxi_duration_model.bin"

        with open(model_file, "wb") as file:
         pickle.dump(model_pipeline, file)

        print(f"Model saved to: {model_file}")

    print("\nTraining pipeline completed successfully.")

    return {
        "training_rmse": train_rmse,
        "validation_rmse": val_rmse,
        "model": model_pipeline,
    }


# --------------------------------------------------
# Command-line entry point
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train an NYC yellow taxi duration "
            "prediction model."
        )
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Training data year, for example 2021.",
    )

    parser.add_argument(
        "--month",
        type=int,
        required=True,
        choices=range(1, 13),
        metavar="MONTH",
        help="Training data month, from 1 to 12.",
    )

    args = parser.parse_args()

    run(
        year=args.year,
        month=args.month,
    )