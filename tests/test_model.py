from pathlib import Path

import joblib
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "random_forest_smote_validated.pkl"
SCALER_PATH = ROOT / "models" / "feature_scaler.pkl"
TEST_DATA_PATH = ROOT / "data" / "processed" / "test_scaled.csv"

TARGET = "Class"

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ==========================================================
# MODEL FILE TESTS
# ==========================================================

def test_model_file_exists():
    assert MODEL_PATH.exists(), (
        f"Model file not found: {MODEL_PATH}"
    )


def test_scaler_file_exists():
    assert SCALER_PATH.exists(), (
        f"Scaler file not found: {SCALER_PATH}"
    )


# ==========================================================
# MODEL LOADING TESTS
# ==========================================================

def test_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)

    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_scaler_can_be_loaded():
    scaler = joblib.load(SCALER_PATH)

    assert scaler is not None
    assert hasattr(scaler, "transform")


# ==========================================================
# DATASET TESTS
# ==========================================================

def test_test_dataset_exists():
    assert TEST_DATA_PATH.exists(), (
        f"Test dataset not found: {TEST_DATA_PATH}"
    )


def test_test_dataset_has_expected_columns():
    df = pd.read_csv(TEST_DATA_PATH)

    expected_columns = FEATURES + [TARGET]

    assert list(df.columns) == expected_columns


def test_test_dataset_has_no_missing_values():
    df = pd.read_csv(TEST_DATA_PATH)

    assert not df.isnull().any().any()


# ==========================================================
# MODEL PREDICTION TEST
# ==========================================================

def test_model_can_predict_test_data():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(TEST_DATA_PATH)

    X = df[FEATURES].head(10)

    predictions = model.predict(X)

    assert len(predictions) == 10
    assert set(predictions).issubset({0, 1})


def test_model_probability_output():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(TEST_DATA_PATH)

    X = df[FEATURES].head(10)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (10, 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()