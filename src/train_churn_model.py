import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score,
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import joblib

# --- Folders ensure ---
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

# --- 1. Load Dataset ---
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# --- 2. Basic Cleaning ---
# TotalCharges ko numeric me convert karo
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# customerID kaam ka nahi, hata do
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# Target ko 0/1 me map karo
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# --- 3. Split X, y ---
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Numeric / Categorical columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# --- 4. Preprocessing Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# --- 5. Random Forest Model ---
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("rf", rf)
])

# --- 6. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 7. Train Model ---
pipeline.fit(X_train, y_train)

# --- 8. Predictions ---
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# --- 9. Metrics ---
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Recall:", rec)
print("ROC-AUC:", auc_score)
print("Confusion Matrix:\n", cm)

# --- 10. Save Model ---
joblib.dump(pipeline, "models/churn_rf_model.pkl")
print("Model saved to models/churn_rf_model.pkl")

# --- 11. Confusion Matrix Plot ---
plt.figure(figsize=(6, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.xticks([0, 1], ["No Churn", "Churn"])
plt.yticks([0, 1], ["No Churn", "Churn"])
plt.tight_layout()
plt.savefig("images/confusion_matrix_rf.png", dpi=300)
plt.close()

# --- 12. ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("images/roc_rf.png", dpi=300)
plt.close()
print("Plots saved in images/")
