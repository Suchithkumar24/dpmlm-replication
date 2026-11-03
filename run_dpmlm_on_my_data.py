"""
Run DP-MLM over your own constructed dataset (CSV of Google Maps reviews).

Expected input CSV: my_reviews_pseudonymised.csv, with columns:
    reviewer_name, review_text, rating

Place my_reviews_pseudonymised.csv in the same folder as this script, then run:
    python run_dpmlm_on_my_data.py

This prints before/after for each review at two epsilon values (shows the
privacy-utility trade-off on YOUR data, not just IMDB) and saves everything
to privatized_my_reviews_pseudonymised.csv for your records / final report later.
"""

import csv
from dpmlm import DPMLM
from dpmlm.utils import calculate_logit_bounds, cleanup

INPUT_FILE = "my_reviews_pseudonymised.csv"
OUTPUT_FILE = "privatized_my_reviews_pseudonymised.csv"
EPSILONS_TO_SHOW = [10, 50, 250]  # same low / medium / high values as the rating study

# ---- Load your dataset ----
print(f"Loading {INPUT_FILE}...")
rows = []
with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)
print(f"Loaded {len(rows)} reviews.\n")

# ---- Set up DP-MLM (reuses the already-cached model if you ran the demo earlier) ----
print("Calibrating logit bounds for RoBERTa-base...")
bounds = calculate_logit_bounds("FacebookAI/roberta-base")
# Use the paper's clipping range (mean, mean + 4 std). The package default flattens the logits,
# which makes epsilon ineffective (see diag_epsilon.py / diag_output.txt).
st = bounds["raw_stats"]
mu, sd = float(st["mean"]), float(st["std"])
bounds = {**bounds, "clip_min": mu, "clip_max": mu + 4 * sd, "sensitivity": 4 * sd}
M = DPMLM(MODEL="FacebookAI/roberta-base", calibration=bounds)

# ---- Privatize each review at each epsilon, print + collect results ----
results = []
print("\n=== Privatizing your constructed dataset ===\n")
for i, row in enumerate(rows):
    name = row["reviewer_name"]
    text = row["review_text"]
    rating = row.get("rating", "")

    print(f"[{i}] Reviewer: {name} (rating: {rating})")
    print(f"  ORIGINAL: {text}")

    result_row = {"reviewer_name": name, "rating": rating, "original_text": text}

    for eps in EPSILONS_TO_SHOW:
        rewritten, *_ = M.dpmlm_rewrite(text, epsilon=eps)  # unpack text, ignore extra metadata
        print(f"  eps={eps:>3}: {rewritten}")
        result_row[f"rewritten_eps{eps}"] = rewritten

    print()
    results.append(result_row)

# ---- Save everything to a CSV ----
print(f"Saving results to {OUTPUT_FILE}...")
fieldnames = ["reviewer_name", "rating", "original_text"] + [f"rewritten_eps{e}" for e in EPSILONS_TO_SHOW]
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

cleanup(model_instances=[M])
print("Done. You now have before/after privatized text for your constructed dataset.")
