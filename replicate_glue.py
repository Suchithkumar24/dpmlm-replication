"""
Usage:
  python replicate_glue.py --task sst2 --n_train 20 --n_val 10 --eps 50 --seeds 1   (timing test)
  python replicate_glue.py --task sst2 --n_train 500 --n_val 200 --seeds 3          (real run)
Rewritten text is cached in ./cache, so re-running does not repeat the slow step.
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch

PAPER = {
    "sst2": {"baseline": 95.72, 10: 68.50, 25: 76.49, 50: 84.10, 100: 86.16, 250: 86.05},
    "cola": {"baseline": 84.72, 10: 69.13, 25: 69.77, 50: 70.85, 100: 70.05, 250: 70.18},
}
MLM = "FacebookAI/roberta-base"


def paper_calibration(bounds):
    """Clip logits to (mean, mean + 4*std), as in the paper (Section 5.1.1).
    The installed dpmlm package's default calibration uses a quantile range that
    sits below the mean and flattens the logits, which makes epsilon ineffective."""
    st = bounds["raw_stats"]
    mu, sd = float(st["mean"]), float(st["std"])
    lo, hi = mu, mu + 4 * sd
    return {**bounds, "clip_min": lo, "clip_max": hi, "sensitivity": hi - lo}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["sst2", "cola", "custom"], default="sst2")
    p.add_argument("--csv", help="for --task custom: CSV with review_text and rating (3 = dropped)")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--n_train", type=int, default=500)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--eps", type=int, nargs="+", default=[10, 25, 50, 100, 250])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--calibration", choices=["paper", "default"], default="paper")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--optim", default="adamw_torch",
                   help="use 'adafactor' if you hit CUDA out-of-memory errors")
    p.add_argument("--classifier", default="microsoft/deberta-v3-base",
                   help="fallback if still out of memory: microsoft/deberta-v3-small (state this in the report)")
    p.add_argument("--out", default="results_replication.csv")
    return p.parse_args()


def load_custom(path, val_frac, seed):
    """Binary sentiment from a CSV: rating >= 4 positive, <= 2 negative, 3 dropped.
    Stratified train/validation split."""
    import pandas as pd
    d = pd.read_csv(path)
    d = d[d["rating"] != 3]
    d = d.assign(label=(d["rating"] >= 4).astype(int))
    tr, va = [], []
    for _, g in d.groupby("label"):
        g = g.sample(frac=1, random_state=seed)
        k = int(round(len(g) * val_frac))
        va.append(g.iloc[:k]); tr.append(g.iloc[k:])
    tr, va = pd.concat(tr).sample(frac=1, random_state=seed), pd.concat(va).sample(frac=1, random_state=seed)
    return ((tr["review_text"].tolist(), tr["label"].tolist()),
            (va["review_text"].tolist(), va["label"].tolist()))


def load_task(task, n_train, n_val, seed):
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", task)
    rng = random.Random(seed)

    def pick(split, n):
        d = ds[split]
        idx = list(range(len(d)))
        rng.shuffle(idx)
        idx = sorted(idx[:n])
        return [d[i]["sentence"] for i in idx], [int(d[i]["label"]) for i in idx]

    return pick("train", n_train), pick("validation", n_val)


def rewrite_all(texts, eps, M, cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)
    out, t0 = [], time.time()
    for i, t in enumerate(texts):
        r = M.dpmlm_rewrite(t, epsilon=eps)
        out.append(r[0] if isinstance(r, tuple) else r)
        if (i + 1) % 25 == 0 or i + 1 == len(texts):
            print(f"    eps={eps}: {i + 1}/{len(texts)} rewritten, {time.time() - t0:.0f}s elapsed", flush=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def similarity(orig, priv):
    """Mean BLEU and mean cosine similarity between original and rewritten text."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from sentence_transformers import SentenceTransformer

    sm = SmoothingFunction().method1
    bleu = np.mean([sentence_bleu([o.split()], p.split(), smoothing_function=sm) for o, p in zip(orig, priv)])
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    a = st.encode(orig, normalize_embeddings=True)
    b = st.encode(priv, normalize_embeddings=True)
    return float(bleu), float((a * b).sum(axis=1).mean())


def finetune_eval(tr_x, tr_y, va_x, va_y, seed, a):
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
    set_seed(seed)
    torch.set_grad_enabled(True)  # the rewriting phase may leave gradients disabled globally
    tok = AutoTokenizer.from_pretrained(a.classifier)
    model = AutoModelForSequenceClassification.from_pretrained(a.classifier, num_labels=2)
    model = model.float()  # checkpoint may load as fp16; mixed-precision training needs fp32 weights

    def enc(x, y):
        d = Dataset.from_dict({"text": x, "label": y})
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=a.max_len), batched=True)

    args = TrainingArguments(
        output_dir="tmp_out", num_train_epochs=a.epochs, per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum, optim=a.optim, per_device_eval_batch_size=16, learning_rate=3e-5, seed=seed, report_to="none",
        save_strategy="no", fp16=torch.cuda.is_available(), disable_tqdm=True, logging_steps=50)
    trainer = Trainer(model=model, args=args, train_dataset=enc(tr_x, tr_y),
                      data_collator=DataCollatorWithPadding(tok))
    model.train()
    trainer.train()
    pred = trainer.predict(enc(va_x, va_y)).predictions.argmax(-1)
    acc = float((pred == np.array(va_y)).mean()) * 100
    del model, trainer
    torch.cuda.empty_cache()
    return acc


def main():
    a = parse_args()
    print(f"GPU available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    os.makedirs("cache", exist_ok=True)
    if a.task == "custom":
        (tr_x, tr_y), (va_x, va_y) = load_custom(a.csv, a.val_frac, a.data_seed)
        name = os.path.splitext(os.path.basename(a.csv))[0]
    else:
        (tr_x, tr_y), (va_x, va_y) = load_task(a.task, a.n_train, a.n_val, a.data_seed)
        name = a.task
    print(f"Task {a.task}: {len(tr_x)} train, {len(va_x)} validation examples")


    from dpmlm import DPMLM
    from dpmlm.utils import calculate_logit_bounds, cleanup, setup_resources
    setup_resources()
    bounds = calculate_logit_bounds(MLM)
    if a.calibration == "paper":
        bounds = paper_calibration(bounds)
    print("Calibration:", a.calibration, {k: bounds[k] for k in ("clip_min", "clip_max", "sensitivity")})
    M = DPMLM(MODEL=MLM, calibration=bounds)
    rewritten = {}
    for eps in a.eps:
        tag = f"cache/{name}_{a.calibration}_eps{eps}_tr{len(tr_x)}_va{len(va_x)}_s{a.data_seed}"
        print(f"Rewriting at epsilon={eps}...")
        rewritten[eps] = (rewrite_all(tr_x, eps, M, tag + "_train.json"),
                          rewrite_all(va_x, eps, M, tag + "_val.json"))
    cleanup(model_instances=[M])  # free GPU memory before fine-tuning


    rows = []
    print("Baseline (original text)...")
    accs = [finetune_eval(tr_x, tr_y, va_x, va_y, s, a) for s in range(a.seeds)]
    rows.append({"task": a.task, "eps": "baseline", "acc_mean": np.mean(accs), "acc_std": np.std(accs),
                 "bleu": 1.0, "cosine": 1.0, "paper_acc": PAPER.get(a.task, {}).get("baseline", "")})
    for eps in a.eps:
        print(f"Fine-tuning on epsilon={eps} text...")
        ptr, pva = rewritten[eps]
        accs = [finetune_eval(ptr, tr_y, pva, va_y, s, a) for s in range(a.seeds)]
        bleu, cos = similarity(va_x, pva)
        rows.append({"task": a.task, "eps": eps, "acc_mean": np.mean(accs), "acc_std": np.std(accs),
                     "bleu": bleu, "cosine": cos, "paper_acc": PAPER.get(a.task, {}).get(eps, "")})

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'eps':>9} | {'mine (mean+/-std)':>20} | {'paper':>7} | {'BLEU':>5} | {'cosine':>6}")
    for r in rows:
        print(f"{str(r['eps']):>9} | {r['acc_mean']:>12.2f} +/- {r['acc_std']:<4.2f} | "
              f"{r['paper_acc']!s:>7} | {r['bleu']:>5.2f} | {r['cosine']:>6.2f}")
    print(f"\nSaved to {a.out}")


if __name__ == "__main__":
    main()
