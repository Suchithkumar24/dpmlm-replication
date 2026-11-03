"""
Replace reviewer names with stable pseudonyms (Reviewer_01, Reviewer_02, ...).

Input CSV columns: reviewer_name, review_text, rating
Writes:
  <out>      pseudonymised CSV, safe to put in the public repo
  <mapping>  name -> pseudonym lookup. KEEP THIS PRIVATE (it is in .gitignore).

Usage:
  python pseudonymise.py my_reviews.csv my_reviews_pseudonymised.csv name_mapping_PRIVATE.csv
"""
import csv
import sys

src, out, mapping = sys.argv[1], sys.argv[2], sys.argv[3]
ids = {}
with open(src, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
for r in rows:
    name = r["reviewer_name"].strip()
    if name not in ids:
        ids[name] = f"Reviewer_{len(ids) + 1:02d}"
    r["reviewer_name"] = ids[name]

with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["reviewer_name", "review_text", "rating"])
    w.writeheader()
    w.writerows(rows)
with open(mapping, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["real_name", "pseudonym"])
    w.writerows(ids.items())
print(f"{len(rows)} reviews, {len(ids)} distinct reviewers -> {out}")
