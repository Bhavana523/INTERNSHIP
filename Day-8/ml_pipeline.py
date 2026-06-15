import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

os.makedirs("visualizations", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")


# 1. LOAD & INSPECT DATA
df = pd.read_csv("dataset.csv")
print("=== Shape ==="); print(df.shape)
print("\n=== First 5 rows ==="); print(df.head())
print("\n=== Data Types ==="); print(df.dtypes)
print("\n=== Descriptive Stats ==="); print(df.describe())

# 2. DATA CLEANING & PREPROCESSING
print("\n=== Missing Values ==="); print(df.isnull().sum())
print("\n=== Duplicate Rows ==="); print(df.duplicated().sum())

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
print("\nAfter cleaning – shape:", df.shape)


# 3. FEATURE / TARGET SPLIT + TRAIN-TEST SPLIT
FEATURES = [
    "Attendance_Percentage", "Assessment_Score",
    "Assignment_Completion_Rate", "Login_Frequency",
    "Study_Hours", "Activity_Count"
]
TARGET = "Final_Completion_Status"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")


# 4. FEATURE SCALING  (fit on train, transform both)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# 5. EDA – CORRELATION HEATMAP
fig, ax = plt.subplots(figsize=(9, 7))
corr = df[FEATURES + [TARGET]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", linewidths=0.5,
    cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
    annot_kws={"size": 10}
)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig("visualizations/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  Saved correlation_heatmap.png")


# 6. MODEL TRAINING
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
}

results = {}
for name, model in models.items():
    # Logistic Regression uses scaled features; tree models use raw
    X_tr = X_train_sc if name == "Logistic Regression" else X_train.values
    X_te = X_test_sc  if name == "Logistic Regression" else X_test.values

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Not Completed", "Completed"]))


# 7. CONFUSION MATRICES
for name, model in models.items():
    X_te = X_test_sc if name == "Logistic Regression" else X_test.values
    y_pred = model.predict(X_te)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Not Completed", "Completed"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix – {name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(f"visualizations/confusion_matrix_{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅  Saved confusion_matrix_{fname}.png")

# 8. MODEL PERFORMANCE COMPARISON BAR CHART
metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
metrics_melt = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(metrics_df))
width = 0.18
metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

for i, (metric, color) in enumerate(zip(metric_cols, colors)):
    scores = metrics_df[metric].values
    bars = ax.bar(x + i * width, scores, width, label=metric, color=color, alpha=0.88)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{score:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics_df["Model"], fontsize=11)
ax.set_ylim(0, 1.08)
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig("visualizations/model_performance_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  Saved model_performance_comparison.png")

# 9. FEATURE IMPORTANCE (Random Forest)
rf_model = models["Random Forest"]
importance_df = pd.DataFrame({
    "Feature":   FEATURES,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(importance_df["Feature"], importance_df["Importance"],
               color=sns.color_palette("viridis", len(FEATURES)))
for bar, val in zip(bars, importance_df["Importance"]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_xlabel("Feature Importance Score", fontsize=12)
ax.set_title("Random Forest – Feature Importance", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("visualizations/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  Saved feature_importance.png")


# 10. PRINT FINAL SUMMARY TABLE
print("\n" + "="*60)
print("  FINAL MODEL PERFORMANCE SUMMARY")
print("="*60)
summary = pd.DataFrame(results).T.map(lambda v: f"{v:.4f}")
print(summary.to_string())

# 11. AT-RISK STUDENT FLAGGING DEMO (predict_proba)
THRESHOLD = 0.40
rf = models["Random Forest"]
proba = rf.predict_proba(X_test.values)[:, 1]
at_risk_mask = proba < THRESHOLD
at_risk_ids = df.iloc[X_test.index[at_risk_mask]]["Student_ID"].values
print(f"\nAt-Risk Students (completion probability < {THRESHOLD}): {at_risk_mask.sum()} students")
print("Sample IDs:", at_risk_ids[:10])
