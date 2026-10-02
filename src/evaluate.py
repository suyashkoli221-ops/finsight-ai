
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
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
CHART_DIR = REPORT_DIR / "phase5_charts"

MODEL_PATH = MODEL_DIR / "random_forest_smote_validated.pkl"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
TARGET = "Class"


# ==========================================================
# LOAD DATA
# ==========================================================

def load_data():
    train_path = DATA_DIR / "train_scaled.csv"
    test_path = DATA_DIR / "test_scaled.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "train_scaled.csv or test_scaled.csv is missing."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].astype(int)

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype(int)

    if list(X.columns) != list(X_test.columns):
        raise ValueError("Training and test features do not match.")

    X_fit, X_val, y_fit, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_val, y_val, X_test, y_test


# ==========================================================
# METRICS
# ==========================================================

def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
    }


# ==========================================================
# THRESHOLD TUNING
# ==========================================================

def tune_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall
    f1_values = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    best_f1 = np.max(f1_values)
    candidates = np.flatnonzero(np.isclose(f1_values, best_f1))

    # When F1 ties, prefer higher precision.
    best_index = candidates[
        np.argmax(precision[candidates])
    ]

    return float(thresholds[best_index])


# ==========================================================
# CHARTS
# ==========================================================

def save_threshold_chart(y_true, probabilities, threshold):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    precision = precision[:-1]
    recall = recall[:-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precision, label="Precision")
    ax.plot(thresholds, recall, label="Recall")
    ax.axvline(
        threshold,
        linestyle="--",
        label=f"Selected threshold: {threshold:.4f}",
    )

    ax.set_title("Validation Threshold Analysis")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        CHART_DIR / "validation_threshold_analysis.png",
        dpi=150,
    )
    plt.close(fig)


def save_confusion_matrix(y_true, predictions):
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title("Final Test Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], labels=["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], labels=["Legitimate", "Fraud"])

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:,}",
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(
        CHART_DIR / "confusion_matrix_test.png",
        dpi=150,
    )
    plt.close(fig)

    return matrix


def save_pr_curve(y_true, probabilities):
    precision, recall, _ = precision_recall_curve(
        y_true, probabilities
    )
    pr_auc = average_precision_score(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        recall,
        precision,
        label=f"Average precision = {pr_auc:.4f}",
    )

    ax.set_title("Final Test Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        CHART_DIR / "precision_recall_test.png",
        dpi=150,
    )
    plt.close(fig)


def save_feature_importance(model, feature_names):
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        REPORT_DIR / "feature_importance.csv",
        index=False,
    )

    top = importance.head(15).sort_values("importance")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(
        CHART_DIR / "feature_importance.png",
        dpi=150,
    )
    plt.close(fig)

    return importance


# ==========================================================
# MAIN
# ==========================================================

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FINSIGHT AI - FINAL MODEL EVALUATION")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Validated model not found: {MODEL_PATH}\n"
            "Run python src/train.py first."
        )

    X_val, y_val, X_test, y_test = load_data()

    print("Validation shape:", X_val.shape)
    print("Final test shape:", X_test.shape)

    model = joblib.load(MODEL_PATH)

    # Tune threshold using validation data only.
    val_probabilities = model.predict_proba(X_val)[:, 1]
    threshold = tune_threshold(y_val, val_probabilities)

    validation_metrics = calculate_metrics(
        y_val,
        val_probabilities,
        threshold,
    )

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    for name, value in validation_metrics.items():
        print(f"{name:12}: {value:.4f}")

    save_threshold_chart(
        y_val,
        val_probabilities,
        threshold,
    )

    # Final evaluation on the untouched test set.
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= threshold
    ).astype(int)

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        threshold,
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    for name, value in test_metrics.items():
        print(f"{name:12}: {value:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    matrix = save_confusion_matrix(y_test, test_predictions)
    save_pr_curve(y_test, test_probabilities)

    importance = save_feature_importance(
        model,
        X_test.columns,
    )

    report = {
        "model": MODEL_PATH.name,
        "threshold_selection": "Maximum validation F1; precision tie-break",
        "selected_threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_confusion_matrix": {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
        "test_rows": int(len(y_test)),
        "test_was_used_for_threshold_selection": False,
    }

    with open(
        REPORT_DIR / "phase5_evaluation.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    with open(
        REPORT_DIR / "selected_threshold.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump({"threshold": threshold}, file, indent=4)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)
    print("Selected threshold:", round(threshold, 4))
    print("Reports saved in:", REPORT_DIR)
    print("Charts saved in:", CHART_DIR)


if __name__ == "__main__":
    main()