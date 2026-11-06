#!/usr/bin/env python3
"""Check the trainable EmbeddingGemma 2 (eg2_train.py) two ways:

1. its embeddings against eg2.py's inference engine, on the GPU (they should agree to cosine 1.000000);
2. its gradients against PyTorch's: the same loss, sum(embeddings * R) for a random R, backpropagated
   through sentence-transformers' model on the CPU in float32, compared weight by weight.

    local/eg2-venv/bin/python models/embeddinggemma2/check_train.py

One text is longer than the 512-token sliding window, so the window's mask is checked too.
"""
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
from eg2_train import EG2Train, nn  # noqa: E402

TEXTS = [
    "hello world",
    "def add(a, b):\n    return a + b",
    "A GPU executes thousands of threads in lockstep groups called wavefronts.",
    inspect.getsource(inspect)[:3000],  # about 700 tokens: past the sliding window
]


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    nn.ctx()
    t = time.time()
    m = EG2Train(train_layers="all", train_ple=True, max_len=1024)
    print(f"loaded in {time.time() - t:.1f} s")
    ids, lens = m.batch(TEXTS, "document")
    print(f"batch: {ids.shape[0]} texts, lengths {list(lens)}, padded to {ids.shape[1]}")

    # 1. against the inference engine
    ref_gpu = EmbeddingGemma2(gpu=nn.ctx().dev, max_tokens=4096, max_seq=1024, verbose=False)
    a = ref_gpu.encode(TEXTS, prompt="document")
    with nn.no_grad():
        b = m(ids, lens).numpy()
    cos = np.sum(a * b, 1)
    print("vs eg2.py (GPU inference): cosine " + ", ".join(f"{c:.6f}" for c in cos))
    del ref_gpu

    # 2. gradients against PyTorch
    R = np.random.default_rng(0).standard_normal((len(TEXTS), m.E)).astype(np.float32)
    t = time.time()
    emb = m(ids, lens)
    loss = nn.sum_all(emb * nn.tensor(R))
    loss.backward()
    nn.ctx().sync()
    print(f"forward + backward on the GPU: {time.time() - t:.2f} s, loss {loss.item():.6f}, "
          f"peak VRAM {nn.ctx().pool.peak >> 20} MB")
    ours = {}
    for name, tns, rng in m._named_weights():
        g = tns.grad.numpy() if tns.grad is not None else None
        if g is not None:
            ours[name] = g[rng[0]:rng[1]] if rng else g

    ref = SentenceTransformer("google/embeddinggemma-2", device="cpu", model_kwargs={"torch_dtype": torch.float32})
    feats = ref.tokenize([PROMPTS["document"] + s for s in TEXTS])
    same = all(list(feats["input_ids"][i][:lens[i]]) == list(ids[i][:lens[i]]) for i in range(len(TEXTS)))
    print(f"token ids identical to sentence-transformers': {same}")
    ref.zero_grad()
    t = time.time()
    out = torch.nn.functional.normalize(ref(feats)["sentence_embedding"], dim=1)
    rloss = (out * torch.as_tensor(R)).sum()
    rloss.backward()
    print(f"PyTorch forward + backward on the CPU: {time.time() - t:.2f} s, loss {rloss.item():.6f}")
    cosr = (out.detach().numpy() * b).sum(1)
    print("vs sentence-transformers: cosine " + ", ".join(f"{c:.6f}" for c in cosr))

    worst, rows = 0.0, []
    for tn, p in ref.named_parameters():
        if p.grad is None:
            continue
        keys = [k for k in ours if tn == k or tn.endswith("." + k)]
        key = max(keys, key=len) if keys else None
        if key is None:
            continue
        g, rg = ours[key], p.grad.numpy()
        if g.shape != rg.shape:
            g = g.reshape(rg.shape)
        e = float(np.abs(g - rg).max() / (np.abs(rg).max() + 1e-12))
        c = float((g * rg).sum() / (np.linalg.norm(g) * np.linalg.norm(rg) + 1e-30))
        rows.append((e, c, key))
        worst = max(worst, e)
    rows.sort(reverse=True)
    print(f"\n{len(rows)} weights compared (of {len(ours)} with gradients). Largest relative differences:")
    for e, c, k in rows[:8]:
        print(f"  {e:.2e}  cosine {c:.7f}  {k}")
    print(f"median relative difference {np.median([r[0] for r in rows]):.2e}")
    ok = worst < 1e-3 and cos.min() > 0.99999 and len(rows) == len(ours)
    print("MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
