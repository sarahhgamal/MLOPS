FROM python:3.14-slim

WORKDIR /app

COPY web-service/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY web-service/predict.py ./web-service/predict.py
COPY models/taxi_duration_model.bin ./models/taxi_duration_model.bin

WORKDIR /app/web-service

EXPOSE 9696

CMD ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]