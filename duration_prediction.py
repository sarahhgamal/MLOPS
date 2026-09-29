from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
import mlflow


# Find the data directory automatically.
script_dir = Path(__file__).resolve().parent

if (script_dir / "yellow_tripdata_2021-01.parquet").exists():
    data_dir = script_dir
elif (script_dir / "data" / "yellow_tripdata_2021-01.parquet").exists():
    data_dir = script_dir / "data"
else:
    raise FileNotFoundError(
        "The parquet files were not found. Make sure they are inside "
        "/workspaces/MLOPS/data."
    )


january_file = data_dir / "yellow_tripdata_2021-01.parquet"
february_file = data_dir / "yellow_tripdata_2021-02.parquet"


def prepare_data(file_path):
    """Load and prepare one yellow taxi dataset."""

    print(f"\nLoading {file_path.name}...")

    df = pd.read_parquet(file_path)

    print(f"Original number of rows: {len(df):,}")

    # Yellow taxi data uses tpep pickup and drop-off columns.
    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"]
    )

    df["tpep_dropoff_datetime"] = pd.to_datetime(
        df["tpep_dropoff_datetime"]
    )

    # Calculate trip duration in minutes.
    df["duration"] = (
        df["tpep_dropoff_datetime"]
        - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # Keep trips between 1 and 60 minutes.
    df = df[
        (df["duration"] >= 1)
        & (df["duration"] <= 60)
    ].copy()

    categorical = [
        "PULocationID",
        "DOLocationID",
    ]

    numerical = [
        "trip_distance",
    ]

    # Convert location IDs to strings so DictVectorizer treats
    # them as categorical values.
    df[categorical] = (
        df[categorical]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    # Replace any missing trip distance with zero.
    df[numerical] = df[numerical].fillna(0)

    print(f"Rows after filtering: {len(df):,}")

    df = df.sample(n=min(200_000, len(df)), random_state=42)
    print(f"Rows used after sampling: {len(df):,}")

    print(f"Average duration: {df['duration'].mean():.2f} minutes")

    return df


# January is training data.
df_train = prepare_data(january_file)

# February is validation data.
df_val = prepare_data(february_file)


# Define the model features.
categorical = [
    "PULocationID",
    "DOLocationID",
]

numerical = [
    "trip_distance",
]


# Convert the selected columns into dictionaries.
train_dicts = df_train[
    categorical + numerical
].to_dict(orient="records")

val_dicts = df_val[
    categorical + numerical
].to_dict(orient="records")


# Convert the dictionaries into feature matrices.
dv = DictVectorizer()

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)


# Create the target arrays.
y_train = df_train["duration"].values
y_val = df_val["duration"].values


print("\nFeature preparation completed.")
print(f"Training feature matrix: {X_train.shape}")
print(f"Validation feature matrix: {X_val.shape}")
print(f"Number of features: {len(dv.feature_names_):,}")


# Train the linear regression model.
print("\nTraining Linear Regression...")

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)


# Generate predictions.
train_predictions = linear_model.predict(X_train)
val_predictions = linear_model.predict(X_val)


# Calculate RMSE.
# Calculate Linear Regression RMSE.
train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        train_predictions,
    )
)

val_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        val_predictions,
    )
)


print("\nLinear Regression results:")
print(f"Training RMSE: {train_rmse:.2f} minutes")
print(f"Validation RMSE: {val_rmse:.2f} minutes")


# Connect this script to the same database used by the MLflow UI.
mlflow.set_tracking_uri(
    "sqlite:////workspaces/MLOPS/mlflow.db"
)
mlflow.set_experiment("nyc-taxi-experiment")


# Record this training run in MLflow.
with mlflow.start_run(run_name="linear-regression"):

    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("training_month", "2021-01")
    mlflow.log_param("validation_month", "2021-02")
    mlflow.log_param("training_sample_size", len(df_train))
    mlflow.log_param("validation_sample_size", len(df_val))

    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("val_rmse", val_rmse)

    mlflow.sklearn.log_model(
        linear_model,
        name="linear-regression-model",
    )


print("\nLinear Regression run saved to MLflow.")

# Train an optional Lasso regression model.
print("\nTraining Lasso Regression...")

lasso_model = Lasso(
    alpha=0.001,
    max_iter=10000,
)

import numpy as np

# Lasso requires sparse matrices with 32-bit index arrays
X_train_lasso = X_train.tocsc(copy=True)
X_val_lasso = X_val.tocsc(copy=True)

X_train_lasso.indices = X_train_lasso.indices.astype(np.int32)
X_train_lasso.indptr = X_train_lasso.indptr.astype(np.int32)

X_val_lasso.indices = X_val_lasso.indices.astype(np.int32)
X_val_lasso.indptr = X_val_lasso.indptr.astype(np.int32)

lasso_model.fit(X_train_lasso, y_train)
lasso_predictions = lasso_model.predict(X_val_lasso)

lasso_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        lasso_predictions,
    )
)


print("\nLasso Regression results:")
print(f"Validation RMSE: {lasso_rmse:.2f} minutes")


# Compare both models.
print("\nModel comparison:")

if val_rmse < lasso_rmse:
    print("Linear Regression produced the lower validation RMSE.")
elif lasso_rmse < val_rmse:
    print("Lasso Regression produced the lower validation RMSE.")
else:
    print("Both models produced the same validation RMSE.")


print("\nTraining completed successfully.")