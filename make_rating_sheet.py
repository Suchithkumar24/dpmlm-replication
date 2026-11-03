"""
Build the human-rating materials from a finished `--task custom` run.

Picks --n validation reviews, assigns each to one epsilon (round robin), and writes:
  rating_sheet_blank.csv   give this to each rater (epsilon NOT shown, so ratings are blind)
  rating_key_PRIVATE.csv   pair_id -> epsilon (do not show raters)
  judge_prompt.txt         the same pairs as one prompt for the LLM-as-judge

Run from the folder that contains ./cache, after replicate_glue.py has finished:
  python make_rating_sheet.py --csv yelp_subset.csv --eps 10 50 250 --n 48
"""
import argparse
import csv
import json
import os
import random

from replicate_glue import load_custom

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--eps", type=int, nargs="+", default=[10, 50, 250])
p.add_argument("--n", type=int, default=48)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--data_seed", type=int, default=42)
p.add_argument("--val_frac", type=float, default=0.2)
p.add_argument("--calibration", default="paper")
a = p.parse_args()

(tr_x, _), (va_x, _) = load_custom(a.csv, a.val_frac, a.data_seed)
name = os.path.splitext(os.path.basename(a.csv))[0]
rewritten = {}
for eps in a.eps:
    f = f"cache/{name}_{a.calibration}_eps{eps}_tr{len(tr_x)}_va{len(va_x)}_s{a.data_seed}_val.json"
    with open(f, encoding="utf-8") as fh:
        rewritten[eps] = json.load(fh)

rng = random.Random(a.seed)
idx = rng.sample(range(len(va_x)), min(a.n, len(va_x)))
pairs = [(i, a.eps[k % len(a.eps)]) for k, i in enumerate(idx)]
rng.shuffle(pairs)

with open("rating_sheet_blank.csv", "w", newline="", encoding="utf-8") as f1, \
     open("rating_key_PRIVATE.csv", "w", newline="", encoding="utf-8") as f2, \
     open("judge_prompt.txt", "w", encoding="utf-8") as f3:
    w1, w2 = csv.writer(f1), csv.writer(f2)
    w1.writerow(["pair_id", "original", "rewritten", "meaning_1to5"])
    w2.writerow(["pair_id", "eps", "val_index"])
    f3.write(
        "You are rating privacy-rewritten text. For each pair, score how well the REWRITTEN text "
        "preserves the meaning of the ORIGINAL review, from 1 to 5:\n"
        "1 = unrelated or nonsense, 2 = a few words of overlap but meaning lost, "
        "3 = partly preserved, 4 = mostly preserved, 5 = same meaning.\n"
        "Judge meaning only, not grammar. Reply ONLY with CSV lines: pair_id,score\n\n")
    for n, (i, eps) in enumerate(pairs, 1):
        pid = f"P{n:02d}"
        w1.writerow([pid, va_x[i], rewritten[eps][i], ""])
        w2.writerow([pid, eps, i])
        f3.write(f"{pid}\nORIGINAL: {va_x[i]}\nREWRITTEN: {rewritten[eps][i]}\n\n")
print(f"{len(pairs)} pairs written (epsilons {a.eps}).")
