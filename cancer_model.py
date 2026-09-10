# ============================================================
# BREAST CANCER PREDICTION - COMPLETE MODEL CODE
# ============================================================
# This file contains the complete ML workflow:
# Data -> Cleaning -> EDA -> Visualization -> Processing ->
# Scaling -> 4 Models -> Evaluation -> Best Model -> Export
#
# IMPORTANT:
# The classification threshold is 50%:
# Probability of Malignant >= 50%  -> Malignant (M)
# Probability of Malignant <  50%  -> Benign (B)
#
# The Streamlit website is a SEPARATE file: cancer_app.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Cancer_Data(1).csv"
MODEL_PATH = BASE_DIR / "cancer_model.joblib"
VIZ_DIR = BASE_DIR / "visualizations"

# Create a folder containing all generated visualizations.
VIZ_DIR.mkdir(exist_ok=True)

# ============================================================
# 2. Load Dataset
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("1. DATASET")
print("=" * 70)
print("Original shape:", df.shape)

# ============================================================
# 3. Data Cleaning
# ============================================================

# Unnamed: 32 is an empty column in this dataset.
df = df.drop(columns=["Unnamed: 32"], errors="ignore")

print("Shape after cleaning:", df.shape)
print("Missing values:", df.isnull().sum().sum())

# ============================================================
# 4. Target Distribution
# ============================================================

print("\nDiagnosis counts:")
print(df["diagnosis"].value_counts())

print("\nDiagnosis percentages:")
print(
    df["diagnosis"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ------------------------------------------------------------
# Visualization 1: Target Distribution
# ------------------------------------------------------------

counts = df["diagnosis"].value_counts()

plt.figure(figsize=(7, 5))
plt.bar(
    ["Benign (B)", "Malignant (M)"],
    [counts.get("B", 0), counts.get("M", 0)]
)
plt.title("Diagnosis Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(VIZ_DIR / "01_diagnosis_distribution.png", dpi=150)
plt.show()
plt.close()

# ============================================================
# 5. Features and Target
# ============================================================

X = df.drop(columns=["id", "diagnosis"])

# B = 0 -> Benign
# M = 1 -> Malignant
y = (df["diagnosis"] == "M").astype(int)

print("\nNumber of samples:", X.shape[0])
print("Number of features:", X.shape[1])

# ============================================================
# 6. Feature Information
# ============================================================

print("\nFeature names:")
for i, feature in enumerate(X.columns, start=1):
    print(f"{i:02d}. {feature}")

# ============================================================
# 7. Feature Correlation
# ============================================================

correlation = X.corr()

# ------------------------------------------------------------
# Visualization 2: Correlation Matrix
# ------------------------------------------------------------

plt.figure(figsize=(15, 11))
plt.imshow(correlation, aspect="auto")
plt.colorbar(label="Correlation")
plt.title("Feature Correlation Matrix")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90,
    fontsize=7
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns,
    fontsize=7
)

plt.tight_layout()
plt.savefig(VIZ_DIR / "02_correlation_matrix.png", dpi=150)
plt.show()
plt.close()

# ============================================================
# 8. Feature Distributions
# ============================================================

# ------------------------------------------------------------
# Visualization 3: First 6 Feature Distributions
# ------------------------------------------------------------

X.iloc[:, :6].hist(
    figsize=(12, 8),
    bins=20
)

plt.suptitle("Feature Distributions - First Six Features")
plt.tight_layout()
plt.savefig(VIZ_DIR / "03_feature_distributions.png", dpi=150)
plt.show()
plt.close()

# ============================================================
# 9. Mean Features: Benign vs Malignant
# ============================================================

mean_features = [
    feature for feature in X.columns
    if feature.endswith("_mean")
]

# ------------------------------------------------------------
# Visualization 4: Mean Feature Comparison
# ------------------------------------------------------------

mean_values = df.groupby("diagnosis")[mean_features].mean().T

plt.figure(figsize=(12, 7))
plt.bar(
    np.arange(len(mean_features)) - 0.2,
    mean_values["B"],
    width=0.4,
    label="Benign"
)
plt.bar(
    np.arange(len(mean_features)) + 0.2,
    mean_values["M"],
    width=0.4,
    label="Malignant"
)

plt.title("Mean Feature Comparison: Benign vs Malignant")
plt.xlabel("Feature")
plt.ylabel("Average Value")
plt.xticks(
    range(len(mean_features)),
    mean_features,
    rotation=90
)
plt.legend()
plt.tight_layout()
plt.savefig(VIZ_DIR / "04_mean_feature_comparison.png", dpi=150)
plt.show()
plt.close()

# ============================================================
# 10. Train / Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ============================================================
# 11. Models + Processing
# ============================================================
# Logistic Regression:
# Imputation -> StandardScaler -> Logistic Regression
#
# SVM:
# Imputation -> StandardScaler -> SVM
#
# Random Forest:
# Imputation -> Random Forest
#
# KNN:
# Imputation -> StandardScaler -> KNN

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            random_state=42
        ))
    ]),

    "Support Vector Machine": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "K-Nearest Neighbors": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=7
        ))
    ])
}

# ============================================================
# 12. Train and Evaluate
# ============================================================

results = {}
trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    trained_models[name] = model

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
    }

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Malignant"],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Visualization 5-8: Confusion Matrix for Each Model
    # --------------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"]
    )

    disp.plot(ax=ax)

    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    plt.savefig(
        VIZ_DIR / f"05_confusion_matrix_{safe_name}.png",
        dpi=150
    )

    plt.show()
    plt.close()

# ============================================================
# 13. Model Comparison
# ============================================================

results_df = (
    pd.DataFrame(results)
    .T
    .sort_values("F1 Score", ascending=False)
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
print(results_df.round(4))

# ------------------------------------------------------------
# Visualization 9: All Metrics
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

x_positions = np.arange(len(results_df.index))
bar_width = 0.20

for i, metric in enumerate(
    ["Accuracy", "Precision", "Recall", "F1 Score"]
):
    plt.bar(
        x_positions + (i - 1.5) * bar_width,
        results_df[metric],
        width=bar_width,
        label=metric
    )

plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0.80, 1.00)
plt.xticks(
    x_positions,
    results_df.index,
    rotation=20
)
plt.legend()
plt.tight_layout()
plt.savefig(VIZ_DIR / "09_model_performance_comparison.png", dpi=150)
plt.show()
plt.close()

# ------------------------------------------------------------
# Visualization 10: Accuracy
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    results_df.index,
    results_df["Accuracy"]
)

plt.title("Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.80, 1.00)
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    VIZ_DIR / "10_accuracy_comparison.png",
    dpi=150
)

plt.show()
plt.close()

# ============================================================
# 14. Best Model
# ============================================================

best_model_name = results_df["F1 Score"].idxmax()
best_model = trained_models[best_model_name]

print("\nBest model based on F1 Score:")
print(best_model_name)

print("\nBest model metrics:")
print(
    results_df.loc[best_model_name].round(4)
)

# ============================================================
# 15. Probability Explanation
# ============================================================
# IMPORTANT:
# predict_proba()[0][1] = probability of Malignant (M)
#
# Threshold = 0.50
#
# Example:
# P(M) = 0.82 -> 82% Malignant -> Malignant
# P(M) = 0.35 -> 35% Malignant -> Benign

print("\n" + "=" * 70)
print("CLASSIFICATION RULE")
print("=" * 70)
print("Malignant probability >= 50% -> Malignant (M)")
print("Malignant probability <  50% -> Benign (B)")

# ============================================================
# 16. Final Training on Full Dataset
# ============================================================

# After evaluation, retrain the selected pipeline using
# all available samples before deployment.
best_model.fit(X, y)

# ============================================================
# 17. Save Deployment Artifact
# ============================================================

feature_ranges = {}

for column in X.columns:

    feature_ranges[column] = {
        "min": float(X[column].min()),
        "max": float(X[column].max()),
        "median": float(X[column].median())
    }

model_artifact = {
    "model": best_model,
    "model_name": best_model_name,
    "feature_names": list(X.columns),
    "feature_ranges": feature_ranges,
    "metrics": results,
    "dataset_shape": df.shape,
    "threshold": 0.50,
    "target_mapping": {
        "B": 0,
        "M": 1
    }
}

joblib.dump(
    model_artifact,
    MODEL_PATH
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)
print("Best model:", best_model_name)
print("Saved to:", MODEL_PATH)
print("Visualizations saved to:", VIZ_DIR)

print("\nNext step:")
print("python -m streamlit run cancer_app.py")
