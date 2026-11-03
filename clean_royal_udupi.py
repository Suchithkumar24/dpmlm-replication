"""
Clean the hand-collected Google Maps reviews of The Royal Udupi (Sydney).

Processing steps (each count is printed and saved to cleaning_log.txt for the report):
  1. drop rows with a missing rating or review text
  2. strip Google's trailing metadata text ("Meal type Lunch...", "Order type Dine in...",
     "Price per person $1-20", "Food: 5 Service: 5..."), which is not review text
  3. drop reviews that are empty or too short after stripping, and exact duplicates
  4. keep reviews of min_words..max_words
  5. replace reviewer names with anonymous IDs (the name -> ID mapping is saved separately
     and must NOT be published)

Usage: python clean_royal_udupi.py the_royal_udupi_reviews.xlsx
Output: royal_udupi_clean.csv (reviewer_name, review_text, rating), name_mapping_royal_udupi_PRIVATE.csv
"""
import argparse
import re

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("xlsx")
p.add_argument("--min_words", type=int, default=10)
p.add_argument("--max_words", type=int, default=60)
a = p.parse_args()

log = []
def note(msg):
    print(msg); log.append(msg)

df = pd.read_excel(a.xlsx)
note(f"raw rows: {len(df)}")
df = df.dropna(subset=["rating", "review_text"]).copy()
note(f"after dropping missing rating/text: {len(df)}")

META_TAIL = re.compile(r"\s*\b(Meal type|Order type|Service type|Price per person|Wait time)\b.*$", re.S)
# Google sometimes cuts its own label off partway ("Meal typ...", "Order t...")
META_CUT = re.compile(r"\s*\b((Meal|Order) t\w*|Price per)\s*[.…]*\s*$")
SCORE_TAIL = re.compile(r"(?:\s*\b(?:Food|Service|Atmosphere): ?\d\b(?!/)…?)+\s*$")
PHOTO_TAIL = re.compile(r"\s*[….\s]*\+\d+\s*$")   # Google's "+2" (extra photos) marker
def strip_meta(s):
    s = str(s).strip()
    for _ in range(3):  # tails can be stacked, e.g. "... Food: 5... +2"
        s = PHOTO_TAIL.sub("", s)
        s = META_TAIL.sub("", s)
        s = META_CUT.sub("", s)
        s = SCORE_TAIL.sub("", s)
        s = s.rstrip(" …").strip()
    return s


df["review_text"] = df["review_text"].map(strip_meta)
df["words"] = df["review_text"].str.split().str.len()
note(f"empty after stripping Google metadata: {int((df.words == 0).sum())}")
df = df[df.words > 0]
before = len(df)
df = df.drop_duplicates("review_text")
note(f"exact duplicates removed: {before - len(df)} -> {len(df)} rows")
df = df[df.words.between(a.min_words, a.max_words)]
note(f"kept {a.min_words}-{a.max_words} words: {len(df)} rows; ratings {df.rating.astype(int).value_counts().sort_index().to_dict()}")

names = {}
ids = []
for n in df["reviewer_name"].fillna("unknown"):
    names.setdefault(n, f"Anonymous_RoyalUdupi_{len(names) + 1:04d}")
    ids.append(names[n])
df["real_name"] = df["reviewer_name"].fillna("unknown")
df["reviewer_name"] = ids
df[["reviewer_name", "review_text", "rating"]].assign(rating=df.rating.astype(int)).to_csv("royal_udupi_clean.csv", index=False)
df[["real_name", "reviewer_name"]].drop_duplicates().to_csv("name_mapping_royal_udupi_PRIVATE.csv", index=False)
note(f"wrote royal_udupi_clean.csv ({len(df)} reviews, {df.reviewer_name.nunique()} reviewers)")
open("cleaning_log.txt", "w", encoding="utf-8").write("\n".join(log) + "\n")
