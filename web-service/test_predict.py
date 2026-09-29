import json

import requests


url = "http://localhost:9696/predict"

ride = {
    "PULocationID": "132",
    "DOLocationID": "45",
    "trip_distance": 2.5,
}

response = requests.post(
    url,
    json=ride,
    timeout=30,
)

print("Status code:", response.status_code)
print(
    "Response:",
    json.dumps(
        response.json(),
        indent=2,
    ),
)