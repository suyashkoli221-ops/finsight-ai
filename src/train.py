
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ==========================================================
# CONFIGURATION
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = DATA_DIR / "model_reports"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
TARGET = "Class"

MODEL_PATH = MODEL_DIR / "random_forest_smote_validated.pkl"


# ==========================================================
# LOAD DATA
# ==========================================================

def load_training_data():
    path = DATA_DIR / "train_scaled.csv"

    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")

    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    if set(y.unique()) != {0, 1}:
        raise ValueError("Expected binary target labels 0 and 1.")

    print("Original training shape:", X.shape)
    print("Original class distribution:")
    print(y.value_counts().sort_index().to_string())

    return X, y


# ==========================================================
# VALIDATION METRICS
# ==========================================================

def evaluate_validation(model, X_val, y_val):
    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    return {
        "threshold": 0.50,
        "accuracy": float(accuracy_score(y_val, predictions)),
        "precision": float(
            precision_score(y_val, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_val, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_val, predictions, zero_division=0)
        ),
        "pr_auc": float(
            average_precision_score(y_val, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_val, probabilities)
        ),
    }


# ==========================================================
# MAIN TRAINING WORKFLOW
# ==========================================================

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("FINSIGHT AI - LEAKAGE-FREE RANDOM FOREST TRAINING")
    print("=" * 65)

    X, y = load_training_data()

    # Split BEFORE applying SMOTE.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nSplit completed:")
    print("Training portion:", X_train.shape)
    print("Validation portion:", X_val.shape)
    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index().to_string())
    print("\nValidation class distribution:")
    print(y_val.value_counts().sort_index().to_string())

    # Apply SMOTE only to the training portion.
    print("\nApplying SMOTE to training data...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train,
        y_train,
    )

    print("Resampled training shape:", X_train_resampled.shape)
    print("Resampled class distribution:")
    print(
        pd.Series(y_train_resampled)
        .value_counts()
        .sort_index()
        .to_string()
    )

    # Train the model using only resampled training data.
    print("\nTraining Random Forest (SMOTE)...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train_resampled, y_train_resampled)

    # Validation is used for diagnostics and later threshold tuning.
    metrics = evaluate_validation(model, X_val, y_val)

    print("\n" + "=" * 65)
    print("VALIDATION RESULTS AT DEFAULT THRESHOLD 0.50")
    print("=" * 65)

    for name, value in metrics.items():
        print(f"{name:12}: {value:.4f}")

    # Save this model under a new name; preserve Phase 4 models.
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model": MODEL_PATH.name,
        "algorithm": "RandomForestClassifier",
        "sampling": "SMOTE applied only to training split",
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "validation_metrics_at_threshold_0.50": metrics,
        "feature_names": list(X.columns),
        "training_rows_before_smote": int(len(X_train)),
        "training_rows_after_smote": int(len(X_train_resampled)),
        "validation_rows": int(len(X_val)),
        "test_data_used_during_training": False,
    }

    metadata_path = REPORT_DIR / "validated_training_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\n" + "=" * 65)
    print("TRAINING COMPLETED")
    print("=" * 65)
    print("Model saved:", MODEL_PATH)
    print("Metadata saved:", metadata_path)
    print("\nThe test dataset was not loaded or used.")
    print("Next: tune the threshold using this validation split.")


if __name__ == "__main__":
    main()