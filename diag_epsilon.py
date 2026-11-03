"""
Run:  python diag_epsilon.py
"""
import numpy as np
from dpmlm import DPMLM
from dpmlm.utils import calculate_logit_bounds, cleanup, setup_resources
from sentence_transformers import SentenceTransformer

MLM = "FacebookAI/roberta-base"
sentences = [
    "the food was absolutely delicious and the service was excellent .",
    "a gripping and beautifully acted film that i loved from start to finish .",
    "this movie was a complete waste of time and money .",
    "the plot is predictable but the performances are charming .",
    "it is a dull , lifeless and forgettable story .",
]

setup_resources()
default_b = calculate_logit_bounds(MLM)
st_ = default_b["raw_stats"]
mu, sd = float(st_["mean"]), float(st_["std"])
paper_b = {**default_b, "clip_min": mu, "clip_max": mu + 4 * sd, "sensitivity": 4 * sd}

st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
orig = st.encode(sentences, normalize_embeddings=True)

for name, b in [("DEFAULT", default_b), ("PAPER (mean, mean+4std)", paper_b)]:
    print(f"\n=== {name}: clip=({b['clip_min']:.2f}, {b['clip_max']:.2f}), sensitivity={b['sensitivity']:.2f}")
    M = DPMLM(MODEL=MLM, calibration=b)
    print("epsilon | mean cosine | example rewrite")
    for eps in [10, 50, 250, 1000]:
        out = []
        for s in sentences:
            r = M.dpmlm_rewrite(s, epsilon=eps)
            out.append(r[0] if isinstance(r, tuple) else r)
        cos = float((orig * st.encode(out, normalize_embeddings=True)).sum(axis=1).mean())
        print(f"{eps:>7} | {cos:>11.3f} | {out[0]}")
    cleanup(model_instances=[M])
