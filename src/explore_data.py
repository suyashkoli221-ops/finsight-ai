
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
CHART_DIR = OUTPUT_DIR / "charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")


# ==========================================
# LOAD DATA
# ==========================================

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download creditcard.csv and place it "
            "inside data/raw/"
        )

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    return df


# ==========================================
# BASIC DATASET INFORMATION
# ==========================================

def inspect_dataset(df):
    print("\n" + "=" * 55)
    print("DATASET OVERVIEW")
    print("=" * 55)

    print("\nDataset shape:")
    print(df.shape)

    print("\nFirst five records:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nDataset information:")
    df.info()

    print("\nStatistical summary:")
    print(df.describe().T)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate records:")
    print(df.duplicated().sum())


# ==========================================
# CLASS DISTRIBUTION
# ==========================================

def analyze_class_distribution(df):
    print("\n" + "=" * 55)
    print("CLASS DISTRIBUTION")
    print("=" * 55)

    counts = df["Class"].value_counts().sort_index()

    percentages = df["Class"].value_counts(
        normalize=True
    ).sort_index() * 100

    summary = pd.DataFrame({
        "Transaction Type": [
            "Legitimate", "Fraud"
        ],
        "Count": [
            counts.get(0, 0),
            counts.get(1, 0)
        ],
        "Percentage": [
            percentages.get(0, 0),
            percentages.get(1, 0)
        ]
    })

    print(summary.to_string(index=False))

    summary.to_csv(
        OUTPUT_DIR / "class_distribution.csv",
        index=False
    )

    plt.figure(figsize=(8, 5))

    ax = sns.countplot(
        data=df,
        x="Class",
        hue="Class",
        legend=False,
        palette={
            0: "#279879",
            1: "#df5656"
        }
    )

    ax.set_title(
        "Legitimate vs Fraudulent Transactions",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel("Transaction Class")
    ax.set_ylabel("Number of Transactions")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Legitimate", "Fraud"])

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")

    plt.tight_layout()
    plt.savefig(
        CHART_DIR / "class_distribution.png",
        dpi=300
    )
    plt.close()


# ==========================================
# TRANSACTION AMOUNT ANALYSIS
# ==========================================

def analyze_amounts(df):
    print("\n" + "=" * 55)
    print("TRANSACTION AMOUNT ANALYSIS")
    print("=" * 55)

    print("\nAmount statistics by class:")

    amount_summary = (
        df.groupby("Class")["Amount"]
        .agg(["count", "mean", "median", "min", "max"])
    )

    print(amount_summary)

    amount_summary.to_csv(
        OUTPUT_DIR / "amount_summary.csv"
    )

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="Class",
        y="Amount",
        hue="Class",
        legend=False,
        showfliers=False,
        palette={
            0: "#279879",
            1: "#df5656"
        }
    )

    plt.title(
        "Transaction Amount Distribution",
        fontsize=15,
        fontweight="bold"
    )
    plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Transaction Amount")
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "amount_distribution.png",
        dpi=300
    )
    plt.close()


# ==========================================
# TRANSACTION TIME ANALYSIS
# ==========================================

def analyze_time(df):
    print("\n" + "=" * 55)
    print("TRANSACTION TIME ANALYSIS")
    print("=" * 55)

    # The dataset time is relative to the
    # first recorded transaction.
    df = df.copy()
    df["Hour"] = (
        (df["Time"] // 3600) % 24
    ).astype(int)

    hourly = (
        df.groupby(["Hour", "Class"])
        .size()
        .reset_index(name="Transactions")
    )

    hourly.to_csv(
        OUTPUT_DIR / "hourly_transactions.csv",
        index=False
    )

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=hourly,
        x="Hour",
        y="Transactions",
        hue="Class",
        style="Class",
        markers=True,
        dashes=False,
        palette={
            0: "#279879",
            1: "#df5656"
        }
    )

    plt.title(
        "Transaction Activity by Relative Hour",
        fontsize=15,
        fontweight="bold"
    )
    plt.xlabel("Relative Hour")
    plt.ylabel("Transaction Count")
    plt.xticks(range(0, 24))
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "hourly_transactions.png",
        dpi=300
    )
    plt.close()


# ==========================================
# CORRELATION ANALYSIS
# ==========================================

def analyze_correlations(df):
    print("\nGenerating correlation heatmap...")

    correlation = df.corr(numeric_only=True)

    correlation.to_csv(
        OUTPUT_DIR / "correlation_matrix.csv"
    )

    plt.figure(figsize=(16, 12))

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        linewidths=0.1,
        cbar_kws={"shrink": 0.8}
    )

    plt.title(
        "Feature Correlation Heatmap",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "correlation_heatmap.png",
        dpi=300
    )
    plt.close()


# ==========================================
# SAVE DATA QUALITY REPORT
# ==========================================

def save_quality_report(df):
    report = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(dtype) for dtype in df.dtypes
        ],
        "Missing Values": [
            int(value) for value in df.isnull().sum()
        ],
        "Unique Values": [
            int(value) for value in df.nunique()
        ]
    })

    report.to_csv(
        OUTPUT_DIR / "data_quality_report.csv",
        index=False
    )

    print("\nData quality report saved.")


# ==========================================
# MAIN
# ==========================================

def main():
    df = load_dataset()

    inspect_dataset(df)

    analyze_class_distribution(df)

    analyze_amounts(df)

    analyze_time(df)

    analyze_correlations(df)

    save_quality_report(df)

    print("\n" + "=" * 55)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 55)

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Charts saved in: {CHART_DIR}")


if __name__ == "__main__":
    main()