# =========================================================
# FILE: app.py
#
# PURPOSE:
# FastAPI application exposing:
#
# 1. /metrics
# 2. /predict
#
# =========================================================

from fastapi import FastAPI
import joblib
import numpy as np
import json

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("model.pkl")

# =========================================================
# LOAD METRICS
# =========================================================

with open("metrics.json") as f:
    metrics = json.load(f)

# =========================================================
# FAST API INIT
# =========================================================

app = FastAPI()

# =========================================================
# ROOT
# =========================================================

@app.get("/")
def home():

    return {
        "message": "MLOps Pipeline Running"
    }

# =========================================================
# METRICS ENDPOINT
#
# Instructor evaluation script will use this
# =========================================================

@app.get("/metrics")
def get_metrics():

    return metrics

# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict(data: dict):

    features = np.array(
        data["features"]
    ).reshape(1, -1)

    prediction = model.predict(features)[0]

    return {
        "prediction": int(prediction)
    }