"""Fine-tune EmbeddingGemma 2 itself for retrieval, contrastively, on (query, document) pairs.

Pairs come from your code index: a function's leading comment or docstring is the query, the code after
it the document (the CodeSearchNet recipe, on your own code). Or bring a JSONL file of {"query",
"document"} lines. The loss is the symmetric InfoNCE of in-batch negatives: each query must pick its own
document out of the batch, and each document its query.

Recall@1, @10 and MRR on held-out pairs (split by file), against every held-out document, are measured
before training and as it goes. The result is a model directory eg2.EmbeddingGemma2(path=...) loads;
vectors from it are not comparable with an index built by the original model (re-index to use it).
"""
import json
import os
import re
import time

import numpy as np

from .codeindex import CodeIndex
from .probe import split_by

META = {
    "name": "Fine-tune EmbeddingGemma 2",
    "group": "On EmbeddingGemma 2",
    "tagline": "Teach the encoder your own retrieval task",
    "description": "Contrastive fine-tuning of the encoder itself: the top layers train, the rest stay frozen, "
                   "and gradient checkpointing keeps a batch in a few GB. Built-in task: find the code a comment "
                   "describes, mined from your own index. Reports recall before and after, and exports a model "
                   "the rest of rawgpu can load.",
    "params": [
        {"key": "source", "label": "Data", "type": "choice", "default": "code index",
         "choices": ["code index", "JSONL file"]},
        {"key": "index", "label": "Code index", "type": "path", "default": "~/rawgpu-index", "advanced": True},
        {"key": "jsonl", "label": "JSONL file", "type": "path", "default": "",
         "help": 'One {"query": "...", "document": "..."} per line.'},
        {"key": "pairs", "label": "Pairs", "type": "int", "default": 20000, "min": 200},
        {"key": "train_layers", "label": "Layers to train", "type": "int", "default": 4, "min": 0, "max": 24,
         "help": "The top N of 24; the projection and final norm always train."},
        {"key": "batch", "label": "Pairs per step", "type": "int", "default": 16, "min": 2, "max": 128},
        {"key": "steps", "label": "Steps", "type": "int", "default": 400, "min": 10},
        {"key": "lr", "label": "Learning rate", "type": "float", "default": 2e-5},
        {"key": "temperature", "label": "Temperature", "type": "float", "default": 0.05, "advanced": True},
        {"key": "doc_len", "label": "Document length", "type": "int", "default": 256, "unit": "tokens",
         "advanced": True},
        {"key": "query_len", "label": "Query length", "type": "int", "default": 64, "unit": "tokens", "advanced": True},
        {"key": "eval_pairs", "label": "Held-out pairs", "type": "int", "default": 500, "advanced": True},
        {"key": "eval_every", "label": "Evaluate every", "type": "int", "default": 100, "unit": "steps",
         "advanced": True},
    ],
    "tiles": [
        {"key": "recall_1", "label": "Recall@1", "fmt": ".1%", "best": "max"},
        {"key": "recall_10", "label": "Recall@10", "fmt": ".1%", "best": "max"},
        {"key": "mrr", "label": "MRR", "fmt": ".3f", "best": "max"},
        {"key": "tok_s", "label": "Throughput", "fmt": ",.0f", "unit": "tokens/s"},
    ],
    "charts": [
        {"title": "Contrastive loss", "series": [{"key": "loss", "label": "Training"}]},
        {"title": "Held-out retrieval", "series": [{"key": "recall_1", "label": "Recall@1"},
                                                   {"key": "recall_10", "label": "Recall@10"}], "fmt": ".0%"},
    ],
    "panels": ["retrieval"],
}

_COMMENT_LANGS = {"C", "C++", "JavaScript", "TypeScript", "Rust", "Swift", "Go", "Java", "Kotlin", "C#", "Objective-C",
                  "OpenCL", "Dart", "PHP", "Solidity", "Shader"}
_DEF = re.compile(r"^\s*(?:pub |export |static |async |public |private |protected |inline |func |fn |function |"
                  r"def |class |struct |impl |interface |enum |[\w:<>\*&\[\], ]+\()")


def _clean(lines):
    out = []
    for l in lines:
        l = re.sub(r"^\s*(/\*\*?|\*/|\*|///?|#)\s?", "", l).rstrip()
        l = l.replace("*/", "").strip()
        if l:
            out.append(l)
    return " ".join(out)


def mine(text, lang):
    """(query, document) from a chunk with a leading comment or docstring before a definition, or None"""
    lines = text.split("\n")
    if lang == "Python":
        m = re.search(r'^(\s*(?:async\s+)?(?:def|class)\s[^\n]*:\s*\n)(\s*)("""|\'\'\')(.*?)\3', text, re.S | re.M)
        if not m:
            return None
        q = " ".join(m.group(4).split())
        doc = (text[:m.start(2)] + text[m.end():]).strip()
    elif lang in _COMMENT_LANGS:
        i = 0
        while i < len(lines) and not lines[i].strip():
            i += 1
        start = i
        if i < len(lines) and lines[i].lstrip().startswith("/*"):
            while i < len(lines) and "*/" not in lines[i]:
                i += 1
            i += 1
        else:
            while i < len(lines) and lines[i].lstrip().startswith("//"):
                i += 1
        if i == start or i >= len(lines) or not _DEF.match(lines[i]):
            return None
        q = _clean(lines[start:i])
        doc = "\n".join(lines[i:]).strip()
    else:
        return None
    q = q.split(". ")[0] if len(q) > 300 else q
    if not (20 <= len(q) <= 400) or doc.count("\n") < 2 or len(doc) < 80:
        return None
    if re.match(r"(?i)^(copyright|license|spdx|todo|fixme|eslint|@ts-|pragma)", q):
        return None
    return q, doc


def load_pairs(run, rng):
    if run.p("source") == "JSONL file":
        path = os.path.expanduser(run.p("jsonl", ""))
        rows = [json.loads(l) for l in open(path) if l.strip()]
        pairs = [(r["query"], r["document"]) for r in rows]
        return pairs, [d for _, d in pairs], f"{path}: {len(pairs):,} pairs"
    ci = CodeIndex(run.p("index"))
    want = int(run.p("pairs"))
    cand = np.flatnonzero((ci.kind == "source code") & np.isin(ci.lang, list(_COMMENT_LANGS | {"Python"})))
    rng.shuffle(cand)
    pairs, keys, seen = [], [], set()
    run.set_status(message="Mining (comment, code) pairs from the index")
    for i in cand:
        r = mine(ci.text(i), ci.lang[i])
        if r and r[0] not in seen:
            seen.add(r[0])
            pairs.append(r)
            keys.append(ci.path[i])
            if len(pairs) >= want:
                break
    return pairs, keys, f"code index: {len(pairs):,} (comment, code) pairs from {len(set(keys)):,} files"


def evaluate(model, qs, ds, run, step, batch=32):
    from rawgpu import nn
    Q, D = [], []
    with nn.no_grad():
        for i in range(0, len(qs), batch):
            Q.append(model(*model.batch(qs[i:i + batch], "code")).numpy())
            D.append(model(*model.batch(ds[i:i + batch], "document")).numpy())
    Q, D = np.concatenate(Q), np.concatenate(D)
    S = Q @ D.T
    rank = (S > S[np.arange(len(S)), np.arange(len(S))][:, None]).sum(1)  # 0 = first
    r = {"recall_1": float((rank < 1).mean()), "recall_5": float((rank < 5).mean()),
         "recall_10": float((rank < 10).mean()), "mrr": float((1.0 / (rank + 1)).mean())}
    hist = np.bincount(np.minimum(rank, 20), minlength=21).tolist()
    worst = np.argsort(-rank)[:3]
    run.event("retrieval", step, pool=len(ds), rank_hist=hist,
              examples=[{"query": qs[i][:200], "rank": int(rank[i]) + 1} for i in worst], **r)
    return r


def train(run):
    from rawgpu import nn
    from eg2_train import EG2Train
    seed = int(run.p("seed", 0))
    rng = np.random.default_rng(seed)
    pairs, keys, what = load_pairs(run, rng)
    split = split_by(keys, seed)
    test = [pairs[i] for i in np.flatnonzero(split == 2)][:int(run.p("eval_pairs"))]
    train_p = [pairs[i] for i in np.flatnonzero(split < 2)]
    if len(train_p) < int(run.p("batch")) or len(test) < 20:
        raise ValueError(f"not enough pairs ({what})")
    run.set_status(message="Loading EmbeddingGemma 2")
    model = EG2Train(train_layers=int(run.p("train_layers")), max_len=int(run.p("doc_len")), verbose=False)
    run.event("info", 0, text=f"{what}; {len(train_p):,} to train on, {len(test)} held out; "
                              f"{model.num_parameters() / 1e6:.1f}M of 271M parameters train")
    total, B, tau = int(run.p("steps")), int(run.p("batch")), float(run.p("temperature"))
    opt = nn.AdamW(model.parameters(), lr=float(run.p("lr")), weight_decay=0.01, max_grad_norm=1.0)
    run.saver = lambda: np.savez(os.path.join(run.dir, "checkpoints", "latest.npz"), **model.trained_weights())
    qs, ds = [q for q, _ in test], [d for _, d in test]
    run.set_status(message="Measuring retrieval before training")
    base = evaluate(model, qs, ds, run, 0)
    run.log(0, **base)
    run.summary(baseline_recall_1=base["recall_1"], baseline_recall_10=base["recall_10"], baseline_mrr=base["mrr"])
    run.set_status(message="")
    labels = np.arange(B)
    qlen = int(run.p("query_len"))
    t_log, toks = time.time(), 0
    for step in run.steps(0, total):
        lr = nn.cosine_lr(step, total, float(run.p("lr")), max(1, total // 20))
        b = rng.choice(len(train_p), B, replace=False)
        qi, ql = model.batch([train_p[i][0] for i in b], "code")
        if qi.shape[1] > qlen:
            qi, ql = qi[:, :qlen], np.minimum(ql, qlen)
        di, dl = model.batch([train_p[i][1] for i in b], "document")
        toks += int(ql.sum() + dl.sum())
        q = model(qi, ql)
        d = model(di, dl)
        loss = nn.scale(nn.add(nn.cross_entropy(nn.matmul(q, d, tb=True, alpha=1 / tau), labels),
                               nn.cross_entropy(nn.matmul(d, q, tb=True, alpha=1 / tau), labels)), 0.5)
        opt.zero_grad()
        loss.backward()
        opt.step(lr)
        if run.every(10) or step == total - 1:
            dt = time.time() - t_log
            run.log(step, loss=loss.item(), lr=lr, grad_norm=opt.grad_norm(), tok_s=toks / dt)
            t_log, toks = time.time(), 0
        if run.every(int(run.p("eval_every"))) or step == total - 1:
            r = evaluate(model, qs, ds, run, step + 1)
            run.log(step, **r)
            run.summary(**r)
            t_log = time.time()
    out = os.path.join(run.dir, "model")
    run.set_status(message="Exporting the model")
    model.export(out)
    run.event("info", total - 1, text=f"Exported a model directory to {out}: "
                                      f"EmbeddingGemma2(path='{out}') loads it (re-index to search with it).")
