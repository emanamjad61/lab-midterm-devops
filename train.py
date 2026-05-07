# =========================================================
# FILE: train.py
#
# PURPOSE:
# 1. Load dataset
# 2. Read model_type from config.json
# 3. Train selected ML model
# 4. Evaluate model
# 5. Save trained model
# 6. Save metrics.json
#
# OUTPUTS:
# - model.pkl
# - metrics.json
#
# =========================================================

import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================================================
# SUPPORTED MODELS
# =========================================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# =========================================================
# LOAD CONFIG
# =========================================================

with open("config.json") as f:
    config = json.load(f)

student_id = config["student_id"]

dataset_version = config["dataset_version"]

model_type = config["model_type"]

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("dataset/train.csv")

# =========================================================
# FEATURES / LABEL
# =========================================================

X = df.drop("label", axis=1)

y = df["label"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# SELECT MODEL FROM CONFIG
# =========================================================

if model_type == "logistic_regression":

    model = LogisticRegression()

elif model_type == "random_forest":

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

elif model_type == "decision_tree":

    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

else:

    raise Exception(
        f"Unsupported model_type: {model_type}"
    )

# =========================================================
# TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)

# =========================================================
# PREDICT
# =========================================================

predictions = model.predict(X_test)

# =========================================================
# EVALUATE
# =========================================================

accuracy = accuracy_score(y_test, predictions)

# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(model, "model.pkl")

# =========================================================
# SAVE METRICS
# =========================================================

metrics = {
    "student_id": student_id,
    "dataset_version": dataset_version,
    "model_type": model_type,
    "accuracy": round(float(accuracy), 4),
    "samples": len(df)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# =========================================================
# SUCCESS LOGS
# =========================================================

print("=" * 50)
print("MODEL TRAINING COMPLETE")
print("=" * 50)

print(f"Student ID       : {student_id}")
print(f"Dataset Version  : {dataset_version}")
print(f"Model Type       : {model_type}")
print(f"Dataset Samples  : {len(df)}")
print(f"Accuracy         : {accuracy:.4f}")

print("=" * 50)