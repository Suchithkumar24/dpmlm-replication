"""

Usage:
  python build_yelp_subset.py path/to/file.csv --n_per_class 250
If column guessing fails, pass --text_col and --stars_col (the script prints the columns).
"""
import argparse

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("csv")
p.add_argument("--n_per_class", type=int, default=250)
p.add_argument("--min_words", type=int, default=10)
p.add_argument("--max_words", type=int, default=60)
p.add_argument("--text_col")
p.add_argument("--stars_col")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", default="yelp_subset.csv")
a = p.parse_args()

df = pd.read_csv(a.csv)
print("Columns:", list(df.columns), "| rows:", len(df))


def guess_text(d):
    cands = [c for c in d.columns if not pd.api.types.is_numeric_dtype(d[c])]
    named = [c for c in cands if "text" in c.lower() or "review" in c.lower()]
    pool = named or cands
    return max(pool, key=lambda c: d[c].astype(str).str.len().mean()) if pool else None


def guess_stars(d):
    for c in d.columns:
        if "star" in c.lower() or "rating" in c.lower():
            v = pd.to_numeric(d[c], errors="coerce").dropna()
            if len(v) and (v == v.round()).all() and v.between(1, 5).all():
                return c
    return None


text_col = a.text_col or guess_text(df)
stars_col = a.stars_col or guess_stars(df)
if text_col is None or stars_col is None or text_col not in df or stars_col not in df:
    raise SystemExit(f"Could not identify columns (text={text_col}, stars={stars_col}). "
                     "Re-run with --text_col NAME --stars_col NAME using the list above.")
print(f"Using text column '{text_col}' and star column '{stars_col}'")

d = df[[text_col, stars_col]].rename(columns={text_col: "review_text", stars_col: "rating"}).dropna()
d["rating"] = pd.to_numeric(d["rating"], errors="coerce")
d = d.dropna().drop_duplicates("review_text")
d = d[~d.review_text.str.contains(r"@|http|www\.", regex=True)]
d["words"] = d.review_text.str.split().str.len()
d = d[d.words.between(a.min_words, a.max_words) & (d.rating != 3)]
d["label"] = np.where(d.rating >= 4, "positive", "negative")
print("After filtering:", d.label.value_counts().to_dict())

d["bin"] = (d.words - a.min_words) // 10
rng = np.random.RandomState(a.seed)
per_bin = {b: min((g.label == "positive").sum(), (g.label == "negative").sum()) for b, g in d.groupby("bin")}
scale = min(1.0, a.n_per_class / max(1, sum(per_bin.values())))
parts = []
for b, g in d.groupby("bin"):
    k = int(round(per_bin[b] * scale))
    for lab in ("positive", "negative"):
        parts.append(g[g.label == lab].sample(k, random_state=rng))
out = pd.concat(parts).sample(frac=1, random_state=rng).reset_index(drop=True)
out.insert(0, "reviewer_name", [f"Anonymous_Yelp_{i + 1:04d}" for i in range(len(out))])
out[["reviewer_name", "review_text", "rating"]].to_csv(a.out, index=False)
print(f"Wrote {len(out)} reviews to {a.out}: {out.label.value_counts().to_dict()}; "
      f"mean words pos/neg = {out[out.label == 'positive'].words.mean():.1f} / {out[out.label == 'negative'].words.mean():.1f}")
