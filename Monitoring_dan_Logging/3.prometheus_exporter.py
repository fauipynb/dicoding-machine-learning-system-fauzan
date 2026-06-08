from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import time
from prometheus_client import Counter, Histogram, make_asgi_app
import os

# 1. Initialize FastAPI
app = FastAPI(title="Mobile Price Prediction API (Dicoding Submission)")

# 2. Prometheus Metrics (Standard Naming)
# Total requests to the API
REQUEST_COUNT = Counter("request_count_total", "Total number of requests received")
# Total successful predictions
PREDICTION_COUNT = Counter("prediction_count_total", "Total number of successful predictions")
# Prediction latency
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Time taken for prediction in seconds")

# 3. Load Model and Scaler
MODEL_PATH = "models/model.joblib"
SCALER_PATH = "src/scaler.joblib"

model = None
scaler = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    print("Warning: Model or Scaler not found.")

# 4. Input Schema
class MobileFeatures(BaseModel):
    battery_power: int
    blue: int
    clock_speed: float
    dual_sim: int
    fc: int
    four_g: int
    int_memory: int
    m_dep: float
    mobile_wt: int
    n_cores: int
    pc: int
    px_height: int
    px_width: int
    ram: int
    sc_h: int
    sc_w: int
    talk_time: int
    three_g: int
    touch_screen: int
    wifi: int

@app.get("/")
def read_root():
    REQUEST_COUNT.inc()
    return {"status": "online", "model": "RandomForest"}

@app.post("/predict")
def predict(features: MobileFeatures):
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    if model is None or scaler is None:
        return {"error": "Model not ready"}

    # Processing
    input_df = pd.DataFrame([features.dict()])
    scaled_input = scaler.transform(input_df)
    
    # Inference
    prediction = model.predict(scaled_input)
    
    # Metrics
    duration = time.time() - start_time
    PREDICTION_LATENCY.observe(duration)
    PREDICTION_COUNT.inc()
    
    return {
        "prediction": int(prediction[0]),
        "latency": duration
    }

# 5. Metrics Endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
