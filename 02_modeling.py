"""
Customer Churn Prediction - Step 2: Modeling & Evaluation
Models: Logistic Regression, Random Forest, XGBoost
Metrics: Accuracy, Recall, ROC-AUC (+ Precision, F1, Confusion Matrix)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix, classification_report)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42

df = pd.read_csv("data/telco_churn_cleaned.csv")

# Drop ID and helper bucket column
df = df.drop(columns=["customerID", "tenure_bucket"], errors="ignore")

y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn"])

num_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
cat_features = [c for c in X.columns if c not in num_features]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
print("Train churn rate:", y_train.mean(), " Test churn rate:", y_test.mean())

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
    ),
}

results = {}
roc_data = {}
fitted_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "accuracy": acc, "recall": rec, "precision": prec, "f1": f1, "roc_auc": auc,
        "confusion_matrix": cm.tolist()
    }
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# Save metrics summary
with open("results/results.json", "w") as f:
    json.dump(results, f, indent=2)

summary_df = pd.DataFrame(results).T[["accuracy", "recall", "precision", "f1", "roc_auc"]]
summary_df = summary_df.sort_values("roc_auc", ascending=False)
print("\n\n=== MODEL COMPARISON ===")
print(summary_df.round(4))
summary_df.to_csv("results/model_comparison.csv")

# ---------- PLOTS ----------

# ROC curves
fig, ax = plt.subplots(figsize=(6, 5))
colors = {"Logistic Regression": "#4C72B0", "Random Forest": "#55A868", "XGBoost": "#C44E52"}
for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colors[name], linewidth=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - Model Comparison")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/07_roc_curves.png")
plt.close()

# Confusion matrices (3 panels)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, res) in zip(axes, results.items()):
    cm = np.array(res["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], cbar=False)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("figures/08_confusion_matrices.png")
plt.close()

# Metric comparison bar chart
fig, ax = plt.subplots(figsize=(8, 5))
summary_df[["accuracy", "recall", "roc_auc"]].plot(kind="bar", ax=ax,
                                                     color=["#4C72B0", "#DD8452", "#55A868"])
ax.set_title("Model Comparison: Accuracy vs Recall vs ROC-AUC")
ax.set_ylabel("Score")
ax.set_xticklabels(summary_df.index, rotation=15)
ax.legend(["Accuracy", "Recall", "ROC-AUC"])
plt.tight_layout()
plt.savefig("figures/09_metric_comparison.png")
plt.close()

# Feature importance for best tree-based model (XGBoost)
xgb_pipe = fitted_pipelines["XGBoost"]
feature_names = xgb_pipe.named_steps["prep"].get_feature_names_out()
importances = xgb_pipe.named_steps["clf"].feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
    "importance", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="#8172B2")
ax.set_title("Top 15 Feature Importances (XGBoost)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("figures/10_feature_importance.png")
plt.close()

print("\nAll plots saved. Modeling complete.")
