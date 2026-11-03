"""

Usage: python audit_dataset.py reviews.xlsx   (or .csv with reviewer_name, review_text, rating)
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

path = sys.argv[1]
df = pd.read_excel(path) if path.endswith("xlsx") else pd.read_csv(path)
df = df[df.rating != 3].copy()
df["y"] = (df.rating >= 4).astype(int)
df["log_words"] = np.log(df.review_text.str.split().str.len())
df["source"] = df.reviewer_name.str.extract(r"Anonymous_(\w+)_", expand=False).fillna("unknown")

def cv(X):
    return cross_val_score(LogisticRegression(max_iter=1000), X, df.y, cv=5).mean() * 100

print(f"{len(df)} reviews (rating 3 dropped): {int(df.y.sum())} positive, {int((1 - df.y).sum())} negative")
print(f"majority class  : {max(df.y.mean(), 1 - df.y.mean()) * 100:.1f}%")
print(f"length only     : {cv(df[['log_words']]):.1f}%")
print(f"source only     : {cv(pd.get_dummies(df.source)):.1f}%")
print("\nrating by source:\n", pd.crosstab(df.source, df.rating).to_string())
print("\nwords per review by source:\n",
      df.assign(w=df.review_text.str.split().str.len()).groupby("source").w.describe()[["mean", "50%", "max"]].round(0).to_string())
