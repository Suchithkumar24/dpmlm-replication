"""
Inter-rater agreement and LLM-judge comparison for the meaning-preservation ratings.

Inputs (CSV with columns pair_id, meaning_1to5; judge file may use 'score'):
  rater1.csv rater2.csv, rating_key_PRIVATE.csv, optional --llm llm_scores.csv
Prints: quadratic-weighted Cohen's kappa, exact / within-1 agreement, Spearman, and the
mean meaning score by epsilon. Saves agreement_results.csv.

  python agreement.py rater1.csv rater2.csv rating_key_PRIVATE.csv --llm llm_scores.csv
"""
import argparse

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

p = argparse.ArgumentParser()
p.add_argument("rater1"); p.add_argument("rater2"); p.add_argument("key")
p.add_argument("--llm")
a = p.parse_args()


def read_any(path):
    """Read a CSV saved by Excel as UTF-8 or as Windows-1252 ('CSV (Comma delimited)'),
    with a comma, semicolon or tab delimiter."""
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            d = pd.read_csv(path, sep=None, engine="python", encoding=enc)
            d.columns = [str(c).strip() for c in d.columns]
            return d
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"{path}: could not decode the file")


def load(path, col):
    d = read_any(path)
    d = d.rename(columns={c: "score" for c in d.columns if c in ("meaning_1to5", "score")})
    if "score" not in d.columns or "pair_id" not in d.columns:
        raise SystemExit(f"{path}: needs columns pair_id and meaning_1to5 (found {list(d.columns)})")
    d = d.dropna(subset=["pair_id"])
    d["pair_id"] = d["pair_id"].astype(str).str.strip()
    raw = d["score"]
    d["score"] = pd.to_numeric(raw.astype(str).str.extract(r"(\d)")[0], errors="coerce")
    bad = d[d["score"].isna() | ~d["score"].between(1, 5)]
    if len(bad):
        print(f"WARNING {path}: {len(bad)} pair(s) have no valid 1-5 score and are ignored: "
              + ", ".join(bad["pair_id"].tolist()))
    d = d.drop(bad.index)
    print(f"{path}: {len(d)} valid scores, columns read: {list(read_any(path).columns)}")
    return d.set_index("pair_id")["score"].astype(int).rename(col)


r1, r2 = load(a.rater1, "r1"), load(a.rater2, "r2")
df = pd.concat([r1, r2], axis=1).dropna()
if df.empty:
    raise SystemExit("No pair has a valid score from BOTH raters. Check the counts above: a file with 0 valid "
                     f"scores is unrated or saved in the wrong format. IDs seen: r1={list(r1.index[:3])}, r2={list(r2.index[:3])}")
key = read_any(a.key).set_index("pair_id")
df = df.join(key["eps"])
if a.llm:
    df = df.join(load(a.llm, "llm"))
df["human_mean"] = df[["r1", "r2"]].mean(axis=1)

rows = []
def add(name, x, y):
    rows.append({"comparison": name,
                 "quadratic_kappa": cohen_kappa_score(x, y, weights="quadratic", labels=[1, 2, 3, 4, 5]),
                 "exact_agree_%": 100 * (x == y).mean(),
                 "within1_%": 100 * ((x - y).abs() <= 1).mean(),
                 "spearman": spearmanr(x, y)[0]})
add("rater1 vs rater2", df.r1, df.r2)
if a.llm:
    d = df.dropna(subset=["llm"]).astype({"llm": int})
    add("LLM vs rater1", d.llm, d.r1)
    add("LLM vs rater2", d.llm, d.r2)
res = pd.DataFrame(rows).round(3)
print(f"{len(df)} pairs\n"); print(res.to_string(index=False))
cols = ["r1", "r2", "human_mean"] + (["llm"] if a.llm else [])
by = df.groupby("eps")[cols].mean().round(2)
print("\nMean meaning score by epsilon:\n", by.to_string())
res.to_csv("agreement_results.csv", index=False)
by.to_csv("meaning_by_epsilon.csv")
