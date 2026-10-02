# FinSight AI – Financial Fraud Detection & Risk Analytics

FinSight AI is a machine learning-based financial fraud detection system that analyzes credit card transactions and identifies potentially fraudulent activity. It uses a Random Forest classifier with SMOTE to address class imbalance and provides an interactive Streamlit dashboard for transaction analysis.

## Features

* **Fraud Detection:** Predicts whether a transaction is legitimate or potentially fraudulent.
* **Machine Learning:** Uses a Random Forest classifier trained with SMOTE.
* **Interactive Dashboard:** Explore model performance and transaction predictions through Streamlit.
* **Single Transaction Analysis:** Enter transaction details to receive a prediction.
* **Batch CSV Analysis:** Upload multiple transactions and download prediction results.
* **Model Evaluation:** Displays classification metrics and performance visualizations.
* **Automated Testing:** Includes tests for model loading, input validation, and prediction functionality.

## Technology Stack

* **Language:** Python
* **Machine Learning:** Scikit-learn, imbalanced-learn
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, Streamlit
* **Model Persistence:** Joblib
* **Testing:** Pytest

## Project Structure

```text
finsight-ai/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   └── processed/
│       ├── train_scaled.csv
│       ├── test_scaled.csv
│       └── model_reports/
│
├── models/
│   ├── feature_scaler.pkl
│   └── random_forest_smote_validated.pkl
│
├── src/
│   ├── preprocessing.py
│   └── train.py
│
├── tests/
│   ├── test_model.py
│   └── test_dashboard.py
│
├── README.md
└── requirements.txt
```

*This is the main project structure; additional files may exist in the repository.*

## Dataset

The project uses the Credit Card Fraud Detection dataset, containing anonymized transaction features.

* **Total transactions:** 284,807
* **Fraudulent transactions:** 492
* **Legitimate transactions:** 284,315
* **Features:** 30 input features
* **Target:** `Class` (0 = legitimate, 1 = fraudulent)

The dataset is highly imbalanced. SMOTE is applied to the training data to help the model learn from the minority fraud class. The test set remains untouched for evaluation.

**Note:** Obtain the dataset separately and place `creditcard.csv` in `data/raw/`. Check the dataset's original source and terms before redistributing it.

## Model Performance

The validated Random Forest model was evaluated on the held-out test set.

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 99.95% |
| Precision | 86.32% |
| Recall    | 83.67% |
| F1 Score  | 84.97% |
| PR-AUC    | 86.15% |
| ROC-AUC   | 97.43% |

### Fraud Detection Results

| Result               | Transactions |
| -------------------- | -----------: |
| Test transactions    |       56,962 |
| Actual fraud cases   |           98 |
| Fraud cases detected |           82 |
| False alarms         |           13 |
| Fraud cases missed   |           16 |

These results are from the project's current evaluation run at a prediction threshold of 0.50. Performance on other datasets or real-world transactions may differ.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/finsight-ai.git
cd finsight-ai
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the downloaded `creditcard.csv` file in:

```text
data/raw/creditcard.csv
```

### 5. Run preprocessing and training

If you need to regenerate the processed data and model artifacts, run:

```bash
python src/preprocessing.py
python src/train.py
```

### 6. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard should open in your browser, usually at:

```text
http://localhost:8501
```

## Running Tests

Run the automated test suite from the project root:

```bash
pytest -v
```

The current test suite contains **18 tests**, which passed in the local project test run.

## Important Notes

* The model's prediction score is an estimate produced by the classifier. It should not be interpreted as a calibrated real-world fraud probability.
* The batch CSV uploader expects the 30 raw input features in the correct format. Do not upload already-scaled test data, because the dashboard applies the saved scaler to `Time` and `Amount`.
* This project is intended for learning, demonstration, and portfolio purposes. It is not a production-ready financial fraud prevention system.

## Future Improvements

* Add threshold tuning based on precision-recall trade-offs.
* Add explainability using feature importance or SHAP.
* Improve dashboard reporting and transaction filtering.
* Add model comparison with other classification algorithms.
* Explore deployment options for a public demo.

## Author

**Suyash Koli**
B.Tech Computer Science and Engineering

## License

A license has not yet been selected for this project. Add a `LICENSE` file before specifying reuse or distribution terms.
