# DP-MLM replication and extension (COMP8240)

Replication of *DP-MLM: Differentially Private Text Rewriting Using Masked Language
Models* (Meisenbacher et al., Findings of ACL 2024; code: https://github.com/sjmeis/DPMLM)
and its application to new review data. The report is submitted separately.

## Files
| File | Purpose |
|---|---|
| `replicate_glue.py` | Rewrites a GLUE subset (`--task sst2`) or a CSV of reviews (`--task custom --csv FILE`) with DP-MLM at several epsilons, fine-tunes DeBERTa-v3-base, reports accuracy, BLEU and cosine similarity |
| `diag_epsilon.py`, `results/diag_output.txt` | Shows that epsilon has no effect under the package's default logit clipping and that the paper's clipping range fixes it |
| `build_yelp_subset.py` | Builds the balanced, length-matched Yelp subset from the Zenodo dataset (doi:10.5281/zenodo.18723813) |
| `audit_dataset.py` | Checks whether review length (or source) predicts sentiment, i.e. whether a dataset has a shortcut |
| `clean_royal_udupi.py`, `results/cleaning_log.txt` | Cleans the hand-collected restaurant reviews and anonymises reviewer names |
| `pseudonymise.py` | Replaces reviewer names with anonymous IDs |
| `make_rating_sheet.py`, `agreement.py` | Builds the 48-pair human rating sheet and computes inter-rater / LLM-judge agreement |
| `ratings/judge_prompt.txt` | Instruction text given to the LLM judge |
| `ratings/llm_scores.csv`, `ratings/rater1_scores_only.csv`, `ratings/rater2_scores_only.csv`, `ratings/rating_key.csv` | Ratings (scores only) and the pair-to-epsilon key |
| `results/results_*.csv`, `results/run*.log` | Raw results and logs behind the report's tables |
| `run_dpmlm_demo.py`, `run_dpmlm_on_my_data.py` | Small demos used for the earlier update presentation |

## Data
- **Yelp subset:** not redistributed (Yelp's terms). Download the zip from
  https://doi.org/10.5281/zenodo.18723813 (Yelp Restaurant Reviews Sentiment Dataset, CC BY 4.0),
  unzip it to get `yelp_sentiment_master_dataset.csv`, and run
  `python build_yelp_subset.py yelp_sentiment_master_dataset.csv --n_per_class 250`.
- **Hand-collected restaurant reviews:** `royal_udupi_clean.csv` (reviewer names replaced by IDs;
  the name mapping is not published).

## Setup
Python 3.10, NVIDIA GPU recommended (developed on an RTX 3050 Ti, 4 GB). Install PyTorch with
CUDA first (https://pytorch.org/get-started/locally/), then:

    python -m pip install -r requirements.txt

## Reproduce
    python replicate_glue.py --task sst2 --n_train 20 --n_val 10 --eps 50 --seeds 1   # quick test
    python replicate_glue.py --task sst2 --n_train 5000 --n_val 872 --epochs 1 --seeds 3
    python replicate_glue.py --task custom --csv yelp_subset.csv --epochs 3 --seeds 3 --max_len 128

Rewritten text is cached in `cache/`. Use `--optim adafactor` if you run out of GPU memory.

## Notes
- The installed `dpmlm` package's default calibration clips logits to a range below the mean,
  which makes epsilon ineffective. `replicate_glue.py` uses the paper's range
  (mean to mean + 4 std) by default (`--calibration paper`).
- Use `python -m pip`, not `pip`, so packages go to the same Python you run scripts with.
