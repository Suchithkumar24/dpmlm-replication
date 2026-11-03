from dpmlm import DPMLM
from dpmlm.utils import setup_resources, calculate_logit_bounds, cleanup

# ---- Step 1: one-time resource setup (downloads NLTK data) ----
print("Setting up resources (first run only)...")
setup_resources()

# ---- Step 2: calibrate and instantiate DP-MLM ----
print("Calibrating logit bounds for RoBERTa-base...")
bounds = calculate_logit_bounds("FacebookAI/roberta-base")
M = DPMLM(MODEL="FacebookAI/roberta-base", calibration=bounds)

# ---- Step 3: rewrite example sentences at the paper's epsilon sweep ----
example_sentences = [
    "The food at this restaurant was absolutely delicious and the service was excellent.",
    "John Smith sent this email to his manager about the quarterly report.",
]

epsilons = [10, 25, 50, 100, 250]

print("\n=== Privatizing example sentences across epsilon values ===\n")
for sentence in example_sentences:
    print(f"ORIGINAL: {sentence}")
    for eps in epsilons:
        rewritten = M.dpmlm_rewrite(sentence, epsilon=eps)
        print(f"  eps={eps:>4}: {rewritten}")
    print()

# ---- Step 4: apply to a small sample of IMDB (new existing dataset) ----
print("=== Loading a small IMDB sample ===")
try:
    from datasets import load_dataset
    imdb = load_dataset("stanfordnlp/imdb", split="train[:5]")  # just 5 reviews for a quick demo

    print("\n=== Privatizing IMDB review snippets (eps=50) ===\n")
    for i, example in enumerate(imdb):
        text = example["text"][:200]  # first 200 chars, full reviews are long
        label = "positive" if example["label"] == 1 else "negative"
        rewritten = M.dpmlm_rewrite(text, epsilon=50)
        print(f"[{i}] Label: {label}")
        print(f"  ORIGINAL : {text}")
        print(f"  REWRITTEN: {rewritten}\n")

except ImportError:
    print("Install the 'datasets' library to run the IMDB portion: pip install datasets")

# ---- Step 5: cleanup ----
cleanup(model_instances=[M])
print("Done.")
