
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.app import (
    prepare_features,
    predict_transactions,
    feature_names,
    model,
    scaler,
)


def make_sample():
    """Create one raw transaction with all required features."""
    return pd.DataFrame(
        [[0.0] + [0.0] * 28 + [100.0]],
        columns=feature_names,
    )


def test_valid_input_has_expected_shape():
    sample = make_sample()
    prepared = prepare_features(sample, feature_names, scaler)

    assert prepared.shape == (1, 30)
    assert list(prepared.columns) == feature_names


def test_missing_feature_is_rejected():
    sample = make_sample().drop(columns=["V1"])

    with pytest.raises(ValueError, match="Missing required features"):
        prepare_features(sample, feature_names, scaler)


def test_extra_feature_is_rejected():
    sample = make_sample()
    sample["Unexpected"] = 10

    with pytest.raises(ValueError, match="Unexpected columns"):
        prepare_features(sample, feature_names, scaler)


def test_missing_values_are_rejected():
    sample = make_sample()
    sample.loc[0, "V1"] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        prepare_features(sample, feature_names, scaler)


def test_infinite_values_are_rejected():
    sample = make_sample()
    sample.loc[0, "Amount"] = float("inf")

    with pytest.raises(ValueError, match="infinite values"):
        prepare_features(sample, feature_names, scaler)


def test_prediction_returns_expected_columns():
    sample = make_sample()
    results = predict_transactions(
        sample, model, scaler, feature_names
    )

    assert len(results) == 1
    assert "Fraud Score" in results.columns
    assert "Prediction" in results.columns
    assert "Risk Level" in results.columns


def test_prediction_score_is_valid():
    sample = make_sample()
    results = predict_transactions(
        sample, model, scaler, feature_names
    )

    score = results["Fraud Score"].iloc[0]

    assert 0 <= score <= 1


def test_prediction_labels_are_valid():
    sample = make_sample()
    results = predict_transactions(
        sample, model, scaler, feature_names
    )

    assert results["Prediction"].iloc[0] in [
        "Fraud",
        "Legitimate",
    ]

    assert results["Risk Level"].iloc[0] in [
        "Low",
        "Medium",
        "High",
    ]


def test_batch_prediction_handles_multiple_rows():
    sample = pd.concat(
        [make_sample(), make_sample(), make_sample()],
        ignore_index=True,
    )

    results = predict_transactions(
        sample, model, scaler, feature_names
    )

    assert len(results) == 3
    assert results["Fraud Score"].notna().all()