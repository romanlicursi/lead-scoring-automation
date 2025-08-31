import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_IN  = Path("data/leads.csv")
DATA_OUT = Path("data/scored_leads.csv")

# ---------- Load ----------
df = pd.read_csv(DATA_IN)

# ---------- Rule-based score ----------
def rule_based_score(row):
    score = 0
    score += min(int(row["Pages_Viewed"]), 20) * 2
    score += min(int(row["CTA_Clicks"]), 20) * 5
    score += float(row["Email_Open_Rate"]) * 20
    if int(row["Demo_Requested"]) == 1:
        score += 15
    if float(row["Annual_Revenue"]) > 1_000_000:
        score += 10
    if row["Lead_Source"] == "Purchased List":
        score -= 15
    if row["Industry"] == "Software":
        score += 10
    return score

df["Raw_Score"] = df.apply(rule_based_score, axis=1)
raw = df["Raw_Score"].to_numpy()
df["Score"] = (raw - raw.min()) / (raw.max() - raw.min()) * 100.0

# ---------- ML score ----------
num = ["Annual_Revenue","Pages_Viewed","Email_Open_Rate","CTA_Clicks","Demo_Requested"]
cat = ["Industry","Lead_Source","Status"]

X = df[num + cat].copy()
y = df["Converted_to_Opportunity"].astype(int).copy()

pre = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat),
])

clf = Pipeline([
    ("pre", pre),
    ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf.fit(X_train, y_train)
auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])

df["Probability"] = clf.predict_proba(X)[:,1]

# ---------- Output ----------
out_cols = ["Lead_ID","Score","Probability"]
DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
df[out_cols].to_csv(DATA_OUT, index=False)

# ---------- Summary ----------
print("Summary")
print("-------")
print(f"Rows: {len(df):,}")
print(f"Mean rule-based Score: {df['Score'].mean():.2f}")
print(f"LogReg AUC (holdout): {auc:.3f}")
print(df[out_cols].sort_values("Score", ascending=False).head())
