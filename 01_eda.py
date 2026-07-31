"""
Customer Churn Prediction - Step 1: Data Loading & EDA
Dataset: IBM Telco Customer Churn (7,043 customers, 21 features)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("data/telco_churn_raw.csv")

print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])

# TotalCharges has blank strings for new customers (tenure=0) -> coerce to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("\nRows with TotalCharges NaN:", df["TotalCharges"].isnull().sum())
print(df[df["TotalCharges"].isnull()][["tenure", "MonthlyCharges", "TotalCharges"]])

# Fill missing TotalCharges (tenure=0 customers) with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

churn_rate = (df["Churn"] == "Yes").mean()
print(f"\nOverall churn rate: {churn_rate:.2%}")

# ---------- EDA PLOTS ----------

# 1. Churn distribution
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["Churn"].value_counts()
ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452"])
for i, v in enumerate(counts.values):
    ax.text(i, v + 30, str(v), ha="center", fontweight="bold")
ax.set_title(f"Churn Distribution (overall rate: {churn_rate:.1%})")
ax.set_ylabel("Number of customers")
plt.tight_layout()
plt.savefig("figures/01_churn_distribution.png")
plt.close()

# 2. Churn rate by contract type
fig, ax = plt.subplots(figsize=(6, 4))
rate_by_contract = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean()).sort_values()
ax.barh(rate_by_contract.index, rate_by_contract.values, color="#C44E52")
for i, v in enumerate(rate_by_contract.values):
    ax.text(v + 0.01, i, f"{v:.1%}", va="center")
ax.set_title("Churn Rate by Contract Type")
ax.set_xlabel("Churn rate")
plt.tight_layout()
plt.savefig("figures/02_churn_by_contract.png")
plt.close()

# 3. Churn rate by tenure buckets
df["tenure_bucket"] = pd.cut(df["tenure"], bins=[0, 12, 24, 36, 48, 60, 72],
                              labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"])
fig, ax = plt.subplots(figsize=(6, 4))
rate_by_tenure = df.groupby("tenure_bucket", observed=True)["Churn"].apply(lambda x: (x == "Yes").mean())
ax.bar(rate_by_tenure.index.astype(str), rate_by_tenure.values, color="#55A868")
ax.set_title("Churn Rate by Tenure (months)")
ax.set_ylabel("Churn rate")
ax.set_xlabel("Tenure bucket (months)")
plt.tight_layout()
plt.savefig("figures/03_churn_by_tenure.png")
plt.close()

# 4. Monthly charges distribution by churn
fig, ax = plt.subplots(figsize=(6, 4))
sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, alpha=0.4, ax=ax)
ax.set_title("Monthly Charges Distribution by Churn")
plt.tight_layout()
plt.savefig("figures/04_monthly_charges_by_churn.png")
plt.close()

# 5. Churn rate by internet service
fig, ax = plt.subplots(figsize=(6, 4))
rate_by_internet = df.groupby("InternetService")["Churn"].apply(lambda x: (x == "Yes").mean()).sort_values()
ax.barh(rate_by_internet.index, rate_by_internet.values, color="#8172B2")
for i, v in enumerate(rate_by_internet.values):
    ax.text(v + 0.01, i, f"{v:.1%}", va="center")
ax.set_title("Churn Rate by Internet Service Type")
ax.set_xlabel("Churn rate")
plt.tight_layout()
plt.savefig("figures/05_churn_by_internet.png")
plt.close()

# 6. Correlation heatmap of numeric features + churn (encoded)
df_corr = df.copy()
df_corr["Churn_bin"] = (df_corr["Churn"] == "Yes").astype(int)
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn_bin"]
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(df_corr[num_cols].corr(), annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
ax.set_title("Correlation: Numeric Features vs Churn")
plt.tight_layout()
plt.savefig("figures/06_correlation_heatmap.png")
plt.close()

print("\nEDA plots saved to figures/")

# Save cleaned df for next step
df.to_csv("data/telco_churn_cleaned.csv", index=False)
print("Cleaned dataset saved.")

# Print some key EDA takeaways to console (used in report)
print("\n--- Key EDA Stats ---")
print("Churn rate month-to-month contract:", rate_by_contract.get("Month-to-month"))
print("Churn rate two year contract:", rate_by_contract.get("Two year"))
print("Churn rate fiber optic:", rate_by_internet.get("Fiber optic"))
print("Churn rate DSL:", rate_by_internet.get("DSL"))
print("Avg tenure churned:", df[df.Churn == "Yes"]["tenure"].mean())
print("Avg tenure retained:", df[df.Churn == "No"]["tenure"].mean())
print("Avg monthly charges churned:", df[df.Churn == "Yes"]["MonthlyCharges"].mean())
print("Avg monthly charges retained:", df[df.Churn == "No"]["MonthlyCharges"].mean())
