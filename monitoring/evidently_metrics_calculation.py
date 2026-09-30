from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2

from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report


PROJECT_DIR = Path(__file__).resolve().parent.parent

REFERENCE_FILE = (
    PROJECT_DIR
    / "monitoring"
    / "data"
    / "reference.parquet"
)

CURRENT_FILE = (
    PROJECT_DIR
    / "monitoring"
    / "data"
    / "current.parquet"
)

REPORT_FILE = (
    PROJECT_DIR
    / "monitoring"
    / "evidently_report.html"
)


def load_data():
    """Load the reference and current datasets."""

    print(f"Loading reference data from: {REFERENCE_FILE}")

    reference_data = pd.read_parquet(
        REFERENCE_FILE
    )

    print(f"Loading current data from: {CURRENT_FILE}")

    current_data = pd.read_parquet(
        CURRENT_FILE
    )

    print(
        f"Reference shape: {reference_data.shape}"
    )

    print(
        f"Current shape: {current_data.shape}"
    )

    return reference_data, current_data


def prepare_monitoring_data(
    reference_data,
    current_data,
):
    """Keep and prepare the monitored columns."""

    monitored_columns = [
        "PULocationID",
        "DOLocationID",
        "trip_distance",
        "actual_duration",
        "predicted_duration",
    ]

    reference_data = reference_data[
        monitored_columns
    ].copy()

    current_data = current_data[
        monitored_columns
    ].copy()

    categorical_columns = [
        "PULocationID",
        "DOLocationID",
    ]

    for column in categorical_columns:
        reference_data[column] = (
            reference_data[column]
            .fillna("-1")
            .astype(str)
        )

        current_data[column] = (
            current_data[column]
            .fillna("-1")
            .astype(str)
        )

    return reference_data, current_data


def create_column_mapping():
    """Tell Evidently what each column means."""

    column_mapping = ColumnMapping()

    column_mapping.target = "actual_duration"

    column_mapping.prediction = (
        "predicted_duration"
    )

    column_mapping.numerical_features = [
        "trip_distance",
    ]

    column_mapping.categorical_features = [
        "PULocationID",
        "DOLocationID",
    ]

    return column_mapping


def calculate_report(
    reference_data,
    current_data,
    column_mapping,
):
    """Calculate the monitoring metrics."""

    report = Report(
        metrics=[
            ColumnDriftMetric(
                column_name="predicted_duration"
            ),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ]
    )

    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    report.save_html(
        str(REPORT_FILE)
    )

    print(
        f"HTML report saved to: {REPORT_FILE}"
    )

    return report


def extract_metrics(report):
    """Extract summary values from the report."""

    report_dictionary = report.as_dict()

    metrics = report_dictionary["metrics"]

    prediction_drift = (
        metrics[0]["result"]["drift_score"]
    )

    number_of_drifted_columns = (
        metrics[1]["result"][
            "number_of_drifted_columns"
        ]
    )

    share_of_missing_values = (
        metrics[2]["result"]["current"][
            "share_of_missing_values"
        ]
    )

    return {
        "prediction_drift": float(
            prediction_drift
        ),
        "number_of_drifted_columns": int(
            number_of_drifted_columns
        ),
        "share_of_missing_values": float(
            share_of_missing_values
        ),
    }


def save_metrics_to_postgres(metrics):
    """Save one monitoring result to PostgreSQL."""

    print(
        "Connecting to the monitoring database..."
    )

    connection = psycopg2.connect(
        host="localhost",
        port=5432,
        database="monitoring",
        user="postgres",
        password="example",
    )

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS
                model_monitoring (
                    id SERIAL PRIMARY KEY,
                    calculation_time TIMESTAMP NOT NULL,
                    reference_period VARCHAR(20)
                        NOT NULL,
                    current_period VARCHAR(20)
                        NOT NULL,
                    model_uri TEXT NOT NULL,
                    prediction_drift
                        DOUBLE PRECISION,
                    number_of_drifted_columns
                        INTEGER,
                    share_of_missing_values
                        DOUBLE PRECISION
                )
                """
            )

            cursor.execute(
                """
                INSERT INTO model_monitoring (
                    calculation_time,
                    reference_period,
                    current_period,
                    model_uri,
                    prediction_drift,
                    number_of_drifted_columns,
                    share_of_missing_values
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                (
                    datetime.now(),
                    "2021-02",
                    "2021-03",
                    (
                        "models:/"
                        "taxi-duration-regressor"
                        "@champion"
                    ),
                    metrics[
                        "prediction_drift"
                    ],
                    metrics[
                        "number_of_drifted_columns"
                    ],
                    metrics[
                        "share_of_missing_values"
                    ],
                ),
            )

        connection.commit()

        print(
            "Monitoring metrics saved "
            "to PostgreSQL."
        )

    finally:
        connection.close()


def main():
    reference_data, current_data = (
        load_data()
    )

    (
        reference_data,
        current_data,
    ) = prepare_monitoring_data(
        reference_data,
        current_data,
    )

    column_mapping = create_column_mapping()

    report = calculate_report(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    metrics = extract_metrics(report)

    print("\nMonitoring results:")

    print(
        "Prediction drift score:",
        metrics["prediction_drift"],
    )

    print(
        "Number of drifted columns:",
        metrics[
            "number_of_drifted_columns"
        ],
    )

    print(
        "Current missing-value share:",
        metrics[
            "share_of_missing_values"
        ],
    )

    save_metrics_to_postgres(metrics)

    print(
        "\nMonitoring calculation "
        "completed successfully."
    )


if __name__ == "__main__":
    main()