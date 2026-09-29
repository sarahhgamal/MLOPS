import subprocess
import sys

from prefect import flow, task


@task(
    name="train-taxi-model",
    retries=0,
)
def run_training_pipeline(year, month):
    command = [
        sys.executable,
        "duration_pipeline.py",
        "--year",
        str(year),
        "--month",
        str(month),
    ]

    print("Running command:")
    print(" ".join(command))

    result = subprocess.run(
        command,
        check=True,
        text=True,
    )

    return result.returncode


@flow(name="monthly-taxi-training")
def taxi_training_flow(year=2021, month=1):
    run_training_pipeline(
        year=year,
        month=month,
    )


if __name__ == "__main__":
    taxi_training_flow(
        year=2021,
        month=1,
    )