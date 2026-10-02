
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

MODEL_DIR = PROJECT_ROOT / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Class"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Only these two features need scaling.
# V1-V28 are already PCA-transformed features.
SCALE_COLUMNS = ["Time", "Amount"]


# ==========================================
# LOAD AND VALIDATE DATA
# ==========================================

def load_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    print("Loading original dataset...")

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Loaded dataset: {df.shape}")

    return df


def validate_data(df):
    print("\nValidating dataset...")

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' is missing."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.isnull().any().any():
        missing = df.isnull().sum()
        raise ValueError(
            "Missing values detected:\n"
            f"{missing[missing > 0]}"
        )

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            "Target must contain only 0 and 1."
        )

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "Both target classes must be present."
        )

    print("Validation completed.")


# ==========================================
# DUPLICATE ANALYSIS
# ==========================================

def analyze_duplicates(df):
    duplicate_count = int(df.duplicated().sum())

    report = {
        "total_rows": int(len(df)),
        "duplicate_rows": duplicate_count,
        "missing_values": int(
            df.isnull().sum().sum()
        )
    }

    print("\nData quality summary:")
    print(json.dumps(report, indent=4))

    with open(
        PROCESSED_DIR / "preprocessing_quality.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    # Do not automatically remove duplicates.
    # Repeated transactions may be meaningful.
    return report


# ==========================================
# FEATURE AND TARGET SEPARATION
# ==========================================

def separate_features(df):
    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].copy()

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in X.dtypes
    ):
        raise ValueError(
            "All features must be numeric."
        )

    print("\nFeatures:", X.shape[1])
    print("Target distribution:")
    print(y.value_counts().sort_index())

    return X, y


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

def split_data(X, y):
    print("\nCreating stratified train/test split...")

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting class distribution:")
    print(y_test.value_counts().sort_index())

    return X_train, X_test, y_train, y_test


# ==========================================
# FEATURE SCALING
# ==========================================

def scale_features(X_train, X_test):
    print("\nScaling Time and Amount...")

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[SCALE_COLUMNS] = scaler.fit_transform(
        X_train[SCALE_COLUMNS]
    )

    X_test[SCALE_COLUMNS] = scaler.transform(
        X_test[SCALE_COLUMNS]
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "feature_scaler.pkl"
    )

    print("Feature scaling completed.")

    return X_train, X_test, scaler


# ==========================================
# HANDLE CLASS IMBALANCE
# ==========================================

def apply_smote(X_train, y_train):
    print("\nApplying SMOTE to training data...")

    print("Before SMOTE:")
    print(y_train.value_counts().sort_index())

    smote = SMOTE(
        random_state=RANDOM_STATE,
        sampling_strategy="auto",
        k_neighbors=5
    )

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    X_resampled = pd.DataFrame(
        X_resampled,
        columns=X_train.columns
    )

    y_resampled = pd.Series(
        y_resampled,
        name=TARGET
    )

    print("\nAfter SMOTE:")
    print(y_resampled.value_counts().sort_index())

    return X_resampled, y_resampled


# ==========================================
# SAVE PROCESSED DATA
# ==========================================

def save_datasets(
    X_train,
    X_test,
    y_train,
    y_test,
    X_smote,
    y_smote
):
    print("\nSaving processed datasets...")

    # Original training set after scaling
    train_data = X_train.copy()
    train_data[TARGET] = y_train

    # Original test set after scaling
    test_data = X_test.copy()
    test_data[TARGET] = y_test

    # Resampled training set
    smote_data = X_smote.copy()
    smote_data[TARGET] = y_smote

    train_data.to_csv(
        PROCESSED_DIR / "train_scaled.csv",
        index=False
    )

    test_data.to_csv(
        PROCESSED_DIR / "test_scaled.csv",
        index=False
    )

    smote_data.to_csv(
        PROCESSED_DIR / "train_smote.csv",
        index=False
    )

    metadata = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "target": TARGET,
        "scaled_features": SCALE_COLUMNS,
        "original_training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "smote_training_rows": int(len(X_smote)),
        "feature_count": int(X_train.shape[1])
    }

    with open(
        PROCESSED_DIR / "preprocessing_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("Datasets saved successfully.")


# ==========================================
# MAIN PIPELINE
# ==========================================

def main():
    print("=" * 55)
    print("FINSIGHT AI - DATA PREPROCESSING")
    print("=" * 55)

    df = load_data()

    validate_data(df)

    analyze_duplicates(df)

    X, y = separate_features(df)

    X_train, X_test, y_train, y_test = split_data(
        X, y
    )

    X_train, X_test, scaler = scale_features(
        X_train, X_test
    )

    X_smote, y_smote = apply_smote(
        X_train, y_train
    )

    save_datasets(
        X_train,
        X_test,
        y_train,
        y_test,
        X_smote,
        y_smote
    )

    print("\n" + "=" * 55)
    print("PREPROCESSING COMPLETED")
    print("=" * 55)

    print("\nNext: Model development and comparison.")


if __name__ == "__main__":
    main()