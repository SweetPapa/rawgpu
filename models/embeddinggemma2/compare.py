#!/usr/bin/env python3
"""Check rawgpu's EmbeddingGemma 2 against the reference (sentence-transformers on the CPU, float32):
the same token ids, and embeddings that agree to cosine ~1. Includes a text of about 1200 tokens and
one of about 7000 (Python's argparse source), to exercise the 512-token sliding window, its 128-token
tiles, and long positions.

  local/eg2-venv/bin/python models/embeddinggemma2/compare.py
"""
import argparse
import inspect
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eg2 import PROMPTS, EmbeddingGemma2  # noqa: E402

TEXTS = [
    "hello world",
    "What is the capital of France?",
    "Paris is the capital and largest city of France, on the Seine.",
    "A GPU executes thousands of threads in lockstep groups called wavefronts.",
    "def add(a, b):\n    return a + b",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "東京は日本の首都です。",
    " ".join(["The Polaris architecture has 36 compute units, each with four SIMD units of sixteen lanes."] * 60),
    inspect.getsource(argparse)[:24000],
]


def main():
    from sentence_transformers import SentenceTransformer
    import torch
    t = time.time()
    ref = SentenceTransformer("google/embeddinggemma-2", device="cpu", model_kwargs={"torch_dtype": torch.float32})
    print(f"reference loaded in {time.time() - t:.1f} s")
    ours = EmbeddingGemma2()
    worst = 1.0
    for prompt in ("query", "document"):
        # token ids first: the reference's tokenizer on the prompted text
        texts = [PROMPTS[prompt] + s for s in TEXTS]
        ref_ids = [ref.tokenizer(s)["input_ids"] for s in texts]
        our_ids = ours.tokenize(TEXTS, prompt)
        same = sum(a == b for a, b in zip(ref_ids, our_ids))
        print(f"[{prompt}] token ids identical for {same}/{len(TEXTS)} texts "
              f"(lengths {', '.join(str(len(i)) for i in our_ids)})")
        r = ref.encode(TEXTS, prompt=PROMPTS[prompt], normalize_embeddings=True, convert_to_numpy=True)
        o = ours.encode(TEXTS, prompt=prompt)
        cos = np.sum(r * o, axis=1)
        worst = min(worst, cos.min())
        print(f"[{prompt}] cosine(ours, reference): " + ", ".join(f"{c:.6f}" for c in cos))
        print(f"[{prompt}] max |difference| of any coordinate: {np.abs(r - o).max():.2e}")
    print("MATCH" if worst > 0.9999 else "MISMATCH", f"(worst cosine {worst:.6f})")


if __name__ == "__main__":
    main()
