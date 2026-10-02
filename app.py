
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================================
# CONFIGURATION
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "random_forest_smote_validated.pkl"
SCALER_PATH = ROOT / "models" / "feature_scaler.pkl"
METADATA_PATH = (
    ROOT / "data" / "processed" / "model_reports"
    / "validated_training_metadata.json"
)
TEST_DATA_PATH = ROOT / "data" / "processed" / "test_scaled.csv"

TARGET = "Class"
SCALE_COLUMNS = ["Time", "Amount"]
THRESHOLD = 0.50

st.set_page_config(
    page_title="FinSight AI | Fraud Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# LOAD MODEL AND SUPPORTING FILES
# ==========================================================

@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    import json

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, scaler, metadata


@st.cache_data
def load_test_data():
    if not TEST_DATA_PATH.exists():
        return None
    return pd.read_csv(TEST_DATA_PATH)


# ==========================================================
# VALIDATION AND PREDICTION
# ==========================================================

def prepare_features(df, feature_names, scaler):
    """Validate feature columns and scale raw transaction inputs."""

    missing = [name for name in feature_names if name not in df.columns]
    extra = [name for name in df.columns if name not in feature_names]

    if missing:
        raise ValueError(
            "Missing required features: " + ", ".join(missing)
        )

    if extra:
        raise ValueError(
            "Unexpected columns: " + ", ".join(extra)
        )

    X = df[feature_names].copy()

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("All transaction features must be numeric.")

    if X.isnull().any().any():
        raise ValueError("Input contains missing values.")

    if not X.map(lambda value: pd.notna(value) and abs(value) != float("inf")).all().all():
        raise ValueError("Input contains infinite values.")

    # Apply the same scaler used during preprocessing.
    X[SCALE_COLUMNS] = scaler.transform(X[SCALE_COLUMNS])

    return X


def predict_transactions(df, model, scaler, feature_names):
    X = prepare_features(df, feature_names, scaler)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    results = df.copy()
    results["Fraud Score"] = probabilities
    results["Prediction"] = [
        "Fraud" if value == 1 else "Legitimate"
        for value in predictions
    ]
    results["Risk Level"] = [
        "High" if value >= 0.70
        else "Medium" if value >= THRESHOLD
        else "Low"
        for value in probabilities
    ]

    return results


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("💳 FinSight AI")
st.sidebar.caption("Financial Fraud Detection & Risk Analytics")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Single Transaction",
        "Batch CSV Analysis",
        "Model Information",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Model: Validated Random Forest + SMOTE")
st.sidebar.caption(f"Decision threshold: {THRESHOLD:.2f}")


# ==========================================================
# LOAD FILES
# ==========================================================

try:
    model, scaler, metadata = load_model_files()
except FileNotFoundError as error:
    st.error(f"Required model file not found: {error}")
    st.info(
        "Check that the model, scaler, and metadata files "
        "are present in their expected folders."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load model files: {error}")
    st.stop()

feature_names = metadata.get("feature_names", [])

if len(feature_names) != 30:
    st.error(
        "Expected 30 model features, but the metadata "
        "does not contain the expected feature list."
    )
    st.stop()


# ==========================================================
# DASHBOARD OVERVIEW
# ==========================================================

if page == "Dashboard Overview":
    st.title("Financial Fraud Analytics Dashboard")
    st.write(
        "Explore model performance and analyze the "
        "transaction dataset."
    )

    test_data = load_test_data()

    if test_data is None:
        st.warning(
            "Test dataset not found. Add "
            "data/processed/test_scaled.csv to view "
            "the overview."
        )
    else:
        if TARGET not in test_data.columns:
            st.error("The test dataset is missing the Class column.")
            st.stop()

        X_test = test_data[feature_names]
        y_test = test_data[TARGET].astype(int)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(int)

        total = len(test_data)
        actual_fraud = int(y_test.sum())
        detected = int(((predictions == 1) & (y_test == 1)).sum())
        false_alarms = int(((predictions == 1) & (y_test == 0)).sum())
        missed = int(((predictions == 0) & (y_test == 1)).sum())

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Test Transactions", f"{total:,}")
        col2.metric("Actual Fraud Cases", f"{actual_fraud:,}")
        col3.metric("Fraud Detected", f"{detected:,}")
        col4.metric("False Alarms", f"{false_alarms:,}")

        st.divider()
        st.subheader("Model Performance")

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            average_precision_score,
            roc_auc_score,
        )

        metrics = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "PR-AUC": average_precision_score(
                y_test, probabilities
            ),
            "ROC-AUC": roc_auc_score(
                y_test, probabilities
            ),
        }

        metric_cols = st.columns(3)
        for index, (name, value) in enumerate(metrics.items()):
            metric_cols[index % 3].metric(
                name, f"{value:.4f}"
            )

        st.caption(
            "Metrics are calculated on the held-out test dataset "
            "at the 0.50 threshold."
        )

        st.divider()
        left, right = st.columns(2)

        with left:
            st.subheader("Actual Transaction Distribution")
            distribution = pd.DataFrame({
                "Class": ["Legitimate", "Fraud"],
                "Count": [
                    int((y_test == 0).sum()),
                    int((y_test == 1).sum()),
                ],
            })

            fig = px.pie(
                distribution,
                names="Class",
                values="Count",
                hole=0.55,
                title="Actual classes in test data",
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Prediction Outcomes")
            outcomes = pd.DataFrame({
                "Outcome": [
                    "Detected fraud",
                    "Missed fraud",
                    "False alarm",
                    "Correct legitimate",
                ],
                "Count": [
                    detected,
                    missed,
                    false_alarms,
                    int(((predictions == 0) & (y_test == 0)).sum()),
                ],
            })

            fig = px.bar(
                outcomes,
                x="Outcome",
                y="Count",
                title="Test prediction outcomes",
                text="Count",
            )
            fig.update_layout(xaxis_title="", yaxis_title="Transactions")
            st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# SINGLE TRANSACTION
# ==========================================================

elif page == "Single Transaction":
    st.title("Single Transaction Fraud Check")
    st.write(
        "Enter all 30 numerical features for one transaction. "
        "For meaningful results, use values prepared in the "
        "same format as the training dataset."
    )

    st.warning(
        "V1–V28 are PCA-transformed features. They are not "
        "ordinary transaction fields such as merchant name "
        "or account number."
    )

    with st.form("single_transaction_form"):
        st.subheader("Transaction Details")

        col1, col2 = st.columns(2)

        with col1:
            time_value = st.number_input(
                "Time (seconds from dataset reference)",
                min_value=0.0,
                value=0.0,
                step=1.0,
            )
            amount_value = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=100.0,
                step=10.0,
            )

        with col2:
            st.caption(
                "The 28 PCA feature values are required "
                "by the trained model."
            )

        st.subheader("PCA Features")

        feature_values = {}
        cols = st.columns(4)

        for index in range(1, 29):
            name = f"V{index}"
            with cols[(index - 1) % 4]:
                feature_values[name] = st.number_input(
                    name,
                    value=0.0,
                    step=0.1,
                    format="%.6f",
                    key=f"single_{name}",
                )

        submitted = st.form_submit_button(
            "Analyze Transaction",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        transaction = {
            "Time": time_value,
            **feature_values,
            "Amount": amount_value,
        }

        input_df = pd.DataFrame([transaction])

        try:
            result = predict_transactions(
                input_df, model, scaler, feature_names
            )

            score = float(result["Fraud Score"].iloc[0])
            prediction = result["Prediction"].iloc[0]
            risk = result["Risk Level"].iloc[0]

            st.divider()
            st.subheader("Analysis Result")

            a, b, c = st.columns(3)
            a.metric("Prediction", prediction)
            b.metric("Fraud Score", f"{score:.2%}")
            c.metric("Risk Level", risk)

            if prediction == "Fraud":
                st.error(
                    "The model classified this transaction "
                    "as potential fraud."
                )
            else:
                st.success(
                    "The model classified this transaction "
                    "as legitimate at the selected threshold."
                )

            st.caption(
                "This is a model output, not a confirmed fraud "
                "determination. Review suspicious transactions "
                "using appropriate verification procedures."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ==========================================================
# BATCH CSV ANALYSIS
# ==========================================================

elif page == "Batch CSV Analysis":
    st.title("Batch Transaction Analysis")
    st.write(
        "Upload a CSV containing raw, unscaled transaction "
        "features. The file must contain all 30 model features."
    )

    st.info(
        "Do not include the Class target column. The dashboard "
        "will apply the saved scaler to Time and Amount."
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            st.write("Uploaded data preview")
            st.dataframe(input_df.head(10), use_container_width=True)

            if TARGET in input_df.columns:
                st.error(
                    "Remove the Class target column before uploading."
                )
            elif st.button(
                "Analyze Transactions",
                type="primary",
            ):
                results = predict_transactions(
                    input_df, model, scaler, feature_names
                )

                fraud_count = int(
                    (results["Prediction"] == "Fraud").sum()
                )
                legitimate_count = len(results) - fraud_count

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Transactions", f"{len(results):,}")
                c2.metric("Potential Fraud", f"{fraud_count:,}")
                c3.metric("Legitimate", f"{legitimate_count:,}")

                st.divider()
                st.subheader("Prediction Distribution")

                distribution = results["Prediction"].value_counts()
                chart_data = pd.DataFrame({
                    "Prediction": distribution.index,
                    "Count": distribution.values,
                })

                fig = px.bar(
                    chart_data,
                    x="Prediction",
                    y="Count",
                    color="Prediction",
                    text="Count",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Detailed Results")
                st.dataframe(
                    results.sort_values(
                        "Fraud Score", ascending=False
                    ),
                    use_container_width=True,
                )

                csv = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Prediction Results",
                    data=csv,
                    file_name="finsight_predictions.csv",
                    mime="text/csv",
                )

        except Exception as error:
            st.error(f"Could not process the uploaded CSV: {error}")


# ==========================================================
# MODEL INFORMATION
# ==========================================================

elif page == "Model Information":
    st.title("Model Information")

    st.subheader("Training Configuration")

    details = {
        "Algorithm": metadata.get(
            "algorithm", "Random Forest"
        ),
        "Model file": metadata.get("model", MODEL_PATH.name),
        "Sampling": metadata.get(
            "sampling", "SMOTE on training split"
        ),
        "Random state": metadata.get("random_state", "Not available"),
        "Validation size": metadata.get(
            "validation_size", "Not available"
        ),
        "Training rows before SMOTE": metadata.get(
            "training_rows_before_smote", "Not available"
        ),
        "Training rows after SMOTE": metadata.get(
            "training_rows_after_smote", "Not available"
        ),
        "Validation rows": metadata.get(
            "validation_rows", "Not available"
        ),
        "Test data used during training": metadata.get(
            "test_data_used_during_training", "Not available"
        ),
        "Prediction threshold": THRESHOLD,
    }

    st.json(details)

    st.subheader("Expected Features")
    st.write(feature_names)

    st.subheader("Validation Metrics")
    validation_metrics = metadata.get(
        "validation_metrics_at_threshold_0.50", {}
    )

    if validation_metrics:
        metrics_df = pd.DataFrame(
            [
                {"Metric": key, "Value": value}
                for key, value in validation_metrics.items()
            ]
        )
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info("Validation metrics were not found in metadata.")

    st.caption(
        "Validation metrics and held-out test metrics are different. "
        "The dashboard's overview calculates test metrics separately."
    )