#!/usr/bin/env python3
"""EmbeddingGemma 2 on the RX 590 (rawgpu) against the same model on the Mac (sentence-transformers,
float32 on the M4 Pro's GPU through MPS), on real retrieval work: MuSiQue's multi-hop questions over
the Wikipedia paragraphs that come with them.

  - corpus: every paragraph of the first --questions questions (20 each, about 120 tokens);
  - retrieval: each question searches the whole corpus; recall@k counts how many of its supporting
    paragraphs (2-4) land in the top k;
  - long documents: each question's 20 paragraphs joined into one document of about 2000 tokens
    (truncated to 2048), to exercise long sequences and the 512-token sliding window.

Each device runs in a process of its own (MPS and a second GPU in one process make MPS run out of
memory), saves its embeddings, and this process compares them.

  local/eg2-venv/bin/python models/embeddinggemma2/bench.py [--questions 500] [--long 64] [--devices rx590,mps]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NAMES = {"rx590": "RX 590 (rawgpu, fp32)", "mps": "M4 Pro GPU (MPS, fp32)"}


def load_musique(n):
    path = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--dgslibisey--MuSiQue/snapshots/*/"
                                        "musique_ans_v1.0_dev.jsonl"))
    if not path:
        from huggingface_hub import hf_hub_download
        path = [hf_hub_download("dgslibisey/MuSiQue", "musique_ans_v1.0_dev.jsonl", repo_type="dataset")]
    rows = [json.loads(l) for l in open(path[0])][:n]
    paras, index, questions, support, longdocs = [], {}, [], [], []
    for r in rows:
        sup = []
        for p in r["paragraphs"]:
            text = p["title"] + ". " + p["paragraph_text"]
            if text not in index:
                index[text] = len(paras)
                paras.append(text)
            if p["is_supporting"]:
                sup.append(index[text])
        questions.append(r["question"])
        support.append(sup)
        longdocs.append("\n\n".join(p["title"] + ". " + p["paragraph_text"] for p in r["paragraphs"]))
    return paras, questions, support, longdocs


def recall(Q, D, support, ks=(1, 5, 10, 20)):
    top = np.argsort(-(Q @ D.T), axis=1)[:, :max(ks)]
    return {k: float(np.mean([len(set(top[i, :k]) & set(s)) / len(s) for i, s in enumerate(support)])) for k in ks}


def timed(fn):
    t = time.time()
    r = fn()
    return r, time.time() - t


def run(device, args):
    """Embed the corpus, the questions and the long documents on one device; save them to args.out."""
    paras, questions, _, longdocs = load_musique(args.questions)
    longdocs = longdocs[:args.long]
    meta = {}
    if device == "rx590":
        from eg2 import EmbeddingGemma2
        model = EmbeddingGemma2(verbose=False)
        enc = lambda texts, p: model.encode(texts, p)
        enc(paras[:64], "document")  # warm-up: compile, first touch
        D, t_d = timed(lambda: enc(paras, "document"))
        Q, t_q = timed(lambda: enc(questions, "query"))
        L, t_l = timed(lambda: enc(longdocs, "document"))
        info = model.gpu.info
        meta = {"engine_mhz": info["engine_mhz"], "temperature_c": info["temperature_c"]}
        ntok = lambda texts, p: sum(len(s) for s in model.tokenize(texts, p))
        meta["tokens"] = [ntok(paras, "document"), ntok(questions, "query"), ntok(longdocs, "document")]
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        from eg2 import PROMPTS
        st = SentenceTransformer("google/embeddinggemma-2", device="mps", model_kwargs={"torch_dtype": torch.float32})
        st.max_seq_length = 2048  # as the RX 590 engine; unbounded otherwise
        # batches as large as MPS's memory allows: 32 short texts, 4 long ones
        enc = lambda texts, p, bs: st.encode(texts, prompt=PROMPTS[p], batch_size=bs, normalize_embeddings=True,
                                             convert_to_numpy=True, show_progress_bar=False)
        enc(paras[:64], "document", 32)
        torch.mps.synchronize()
        D, t_d = timed(lambda: enc(paras, "document", 32))
        Q, t_q = timed(lambda: enc(questions, "query", 32))
        L, t_l = timed(lambda: enc(longdocs, "document", 4))
    np.savez(args.out, D=D, Q=Q, L=L, times=np.array([t_d, t_q, t_l]), meta=json.dumps(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=500)
    ap.add_argument("--long", type=int, default=64, help="long documents to embed")
    ap.add_argument("--devices", default="rx590,mps", help="comma-separated: rx590, mps")
    ap.add_argument("--run", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.run:
        return run(args.run, args)

    paras, questions, support, longdocs = load_musique(args.questions)
    longdocs = longdocs[:args.long]
    print(f"MuSiQue dev: {len(questions)} questions, {len(paras)} paragraphs, {len(longdocs)} long documents\n")
    results, tokens = {}, None
    print(f"{'':<24} {'paragraphs':>22} {'questions':>20} {'long docs':>20}   retrieval recall@1/5/10/20", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        for dev in args.devices.split(","):
            out = os.path.join(tmp, dev + ".npz")
            subprocess.run([sys.executable, os.path.abspath(__file__), "--run", dev, "--out", out,
                            "--questions", str(args.questions), "--long", str(args.long)],
                           check=True, stderr=subprocess.DEVNULL)
            z = np.load(out)
            meta = json.loads(str(z["meta"]))
            tokens = meta.get("tokens", tokens)
            results[dev] = (z["D"], z["Q"], z["L"], *z["times"])
            D, Q, L, td, tq, tl = results[dev]
            rc = recall(Q, D, support)
            tok = tokens or [0, 0, 0]
            print(f"{NAMES[dev]:<24} {len(paras) / td:7.0f}/s {tok[0] / td / 1e3:6.1f}k tok/s "
                  f"{len(questions) / tq:7.0f}/s {tok[1] / tq / 1e3:5.1f}k tok/s "
                  f"{len(longdocs) / tl:6.1f}/s {tok[2] / tl / 1e3:5.1f}k tok/s   "
                  + " / ".join(f"{rc[k]:.3f}" for k in sorted(rc)), flush=True)
            if "engine_mhz" in meta:
                print(f"{'':<24} (engine {meta['engine_mhz']} MHz, {meta['temperature_c']} C at the end)", flush=True)
    if tokens:
        print(f"\ntokens: paragraphs {tokens[0]} (avg {tokens[0] / len(paras):.0f}), questions {tokens[1]} "
              f"(avg {tokens[1] / len(questions):.0f}), long documents {tokens[2]} (avg {tokens[2] / len(longdocs):.0f})")
    if "rx590" in results and "mps" in results:
        a, b = results["rx590"], results["mps"]
        cos = [np.sum(x * y, axis=1) for x, y in zip(a[:3], b[:3])]
        print(f"agreement, cosine(RX 590, M4 Pro) per embedding: paragraphs min {cos[0].min():.6f}, "
              f"questions min {cos[1].min():.6f}, long documents min {cos[2].min():.6f}")
        ta, tb = sum(a[3:]), sum(b[3:])
        print(f"total time: RX 590 {ta:.1f} s, M4 Pro {tb:.1f} s (the RX 590 at {tb / ta:.2f}x the M4 Pro's speed)")
    D, Q = next(iter(results.values()))[:2]
    i = 3
    print(f"\nexample: \"{questions[i]}\"")
    for j in np.argsort(-(Q[i] @ D.T))[:3]:
        print(f"  {'*' if j in support[i] else ' '} {paras[j][:110]}...")
    print("  (* = a supporting paragraph)")


if __name__ == "__main__":
    main()
