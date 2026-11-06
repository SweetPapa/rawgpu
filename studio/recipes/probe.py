"""Salience filters: small probes on EmbeddingGemma 2's embeddings that say whether an input is about something.

The diagram's "270M probes, cosine": EmbeddingGemma 2 (270M text parameters) embeds the input once, and
each filter is a few hundred numbers on top. Two kinds, compared on held-out data:

    cosine   similarity to the mean embedding of the concept's examples (no training at all)
    probe    a trained logistic-regression (or one-hidden-layer) head, calibrated probabilities

Data: your code index (concepts from each chunk's path: file kinds, languages or projects), or a JSONL file
of {"text": ..., "labels": [...]} lines. Splits are by file, so near-duplicate chunks cannot leak.
"""
import hashlib
import json
import os
import time

import numpy as np

from . import metrics
from .codeindex import CodeIndex, KINDS

META = {
    "name": "Salience probes",
    "group": "On EmbeddingGemma 2",
    "tagline": "Cheap filters on embeddings: is this about X?",
    "description": "One small detector per concept on top of EmbeddingGemma 2's embeddings, against the "
                   "no-training baseline of cosine similarity to a prototype. Trains in seconds from your code "
                   "index (file kinds, languages or projects) or your own labeled JSONL. Reports AUC, F1 and "
                   "calibration per concept, and saves the filters for reuse.",
    "params": [
        {"key": "source", "label": "Data", "type": "choice", "default": "code index",
         "choices": ["code index", "JSONL file"]},
        {"key": "concepts", "label": "Concepts", "type": "choice", "default": "file kinds",
         "choices": ["file kinds", "languages", "projects"], "help": "From the code index: what each filter detects."},
        {"key": "embeddings", "label": "Embeddings", "type": "choice", "default": "content only",
         "choices": ["content only", "as indexed (with the file path)"],
         "help": "The index embedded each chunk with its path as a title, and these labels come from paths: "
                 "\"content only\" re-embeds the text alone on the card (cached after the first time)."},
        {"key": "index", "label": "Code index", "type": "path", "default": "~/rawgpu-index", "advanced": True},
        {"key": "jsonl", "label": "JSONL file", "type": "path", "default": "",
         "help": 'One {"text": "...", "labels": ["a", "b"]} per line; texts are embedded on the card.'},
        {"key": "samples", "label": "Examples", "type": "int", "default": 10000, "min": 1000, "max": 435000},
        {"key": "hidden", "label": "Hidden units", "type": "choice", "default": 0, "choices": [0, 64, 256],
         "help": "0: a linear probe (logistic regression)."},
        {"key": "steps", "label": "Steps", "type": "int", "default": 1500, "min": 10},
        {"key": "lr", "label": "Learning rate", "type": "float", "default": 1e-2},
        {"key": "batch", "label": "Batch", "type": "int", "default": 1024, "advanced": True},
    ],
    "tiles": [
        {"key": "auc_probe", "label": "Probe AUC (mean)", "fmt": ".3f", "best": "max"},
        {"key": "auc_cosine", "label": "Cosine AUC (mean)", "fmt": ".3f", "best": "max"},
        {"key": "f1_probe", "label": "Probe F1 (mean)", "fmt": ".3f", "best": "max"},
        {"key": "ece_probe", "label": "Calibration error", "fmt": ".3f", "best": "min", "help": "0 is perfect"},
    ],
    "charts": [
        {"title": "Loss", "series": [{"key": "loss", "label": "Training"}, {"key": "val_loss", "label": "Validation"}]},
        {"title": "Validation AUC", "series": [{"key": "val_auc", "label": "Probe"}], "small": True},
    ],
    "panels": ["probes"],
}


def split_by(keys, seed):
    """0 train, 1 validation, 2 test (80/10/10), by a hash of each key (a file path)"""
    h = np.array([int(hashlib.md5(f"{seed}:{k}".encode()).hexdigest()[:8], 16) % 10 for k in keys])
    return np.where(h < 8, 0, np.where(h == 8, 1, 2))


def eg2_encoder():
    from rawgpu import nn
    from eg2 import EmbeddingGemma2
    return EmbeddingGemma2(gpu=nn.ctx().dev, max_tokens=8192, max_seq=2048, verbose=False)


def load_data(run, rng):
    """X [n, 768] unit vectors, Y [n, K] 0/1, concept names and descriptions, split [n]"""
    if run.p("source") == "JSONL file":
        path = os.path.expanduser(run.p("jsonl", ""))
        rows = [json.loads(l) for l in open(path) if l.strip()]
        texts = [r["text"] for r in rows]
        names = sorted({l for r in rows for l in r.get("labels", [])})
        Y = np.array([[n in r.get("labels", []) for n in names] for r in rows], np.float32)
        run.set_status(message=f"Embedding {len(texts):,} texts on the card")
        enc = eg2_encoder()
        X = enc.encode(texts, prompt="document")
        del enc
        split = split_by(texts, int(run.p("seed", 0)))
        return X, Y, names, {}, split, f"{path}: {len(rows):,} texts, {len(names)} labels"
    run.set_status(message="Reading the code index")
    ci = CodeIndex(run.p("index"))
    n = min(int(run.p("samples")), len(ci))
    idx = np.sort(rng.choice(len(ci), n, replace=False))
    concepts = run.p("concepts")
    if concepts == "file kinds":
        names, col = list(KINDS), ci.kind
        desc = KINDS
    elif concepts == "languages":
        names, col, desc = ci.top(ci.lang, 8), ci.lang, {}
    else:
        names, col, desc = ci.top(ci.project, 8, exclude=()), ci.project, {}
    vals = col[idx]
    Y = np.stack([(vals == k) for k in names], 1).astype(np.float32)
    content = run.p("embeddings", "content only") == "content only"
    X = ci.content_vectors(idx, lambda m: run.set_status(message=m)) if content else ci.vectors(idx)
    split = split_by([ci.path[i] for i in idx], int(run.p("seed", 0)))
    return X, Y, names, desc, split, (f"code index: {n:,} of {len(ci):,} chunks, {concepts}, "
                                      f"{'text only' if content else 'embedded with their paths'}")


def build(D, K, hidden, rng):
    from rawgpu import nn
    return nn.MLP([D, int(hidden), K], act="gelu", rng=rng) if int(hidden) else nn.Linear(D, K, rng=rng)


def predict(model, X, batch=8192):
    """Logits; the unit-length embeddings scaled by sqrt(dim), so their coordinates have about unit variance
    (a probe on raw unit vectors needs large weights and trains slowly)"""
    from rawgpu import nn
    out, s = [], np.float32(np.sqrt(X.shape[1]))
    with nn.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(nn.tensor(X[i:i + batch] * s)).numpy())
    return np.concatenate(out)


def train(run):
    from rawgpu import nn
    seed = int(run.p("seed", 0))
    rng = np.random.default_rng(seed)
    X, Y, names, desc, split, what = load_data(run, rng)
    K, D = Y.shape[1], X.shape[1]
    tr, va, te = (split == 0), (split == 1), (split == 2)
    if min(tr.sum(), va.sum(), te.sum()) == 0:
        raise ValueError(f"too few examples to split ({len(X)}: {tr.sum()} train, {va.sum()} validation, {te.sum()} "
                         f"test); texts are split 80/10/10, so use at least a few dozen")
    run.event("info", 0, text=f"{what}; {tr.sum():,} train / {va.sum():,} validation / {te.sum():,} test; "
                              f"{K} concepts: {', '.join(names)}")
    model = build(D, K, run.p("hidden", 0), rng)
    total, B = int(run.p("steps")), int(run.p("batch"))
    opt = nn.AdamW(model.parameters(), lr=float(run.p("lr")), weight_decay=1e-4)
    Xtr, Ytr = X[tr], Y[tr]
    Xtr_s = Xtr * np.float32(np.sqrt(D))
    run.saver = lambda: None
    run.set_status(message="")
    t0 = time.time()
    for step in run.steps(0, total):
        lr = nn.cosine_lr(step, total, float(run.p("lr")), 20)
        b = rng.integers(0, len(Xtr), B)
        loss = nn.bce_with_logits(model(nn.tensor(Xtr_s[b])), Ytr[b])
        opt.zero_grad()
        loss.backward()
        opt.step(lr)
        if run.every(max(1, total // 50)) or step == total - 1:
            lv = predict(model, X[va])
            pv = 1 / (1 + np.exp(-lv))
            vl = float(-(Y[va] * np.log(np.maximum(pv, 1e-7)) + (1 - Y[va]) * np.log(np.maximum(1 - pv, 1e-7))).mean())
            aucs = [metrics.auc(lv[:, k], Y[va][:, k]) for k in range(K)]
            run.log(step, loss=loss.item(), val_loss=vl, val_auc=float(np.mean([a for a in aucs if a is not None])),
                    lr=lr, examples_s=(step + 1) * B / (time.time() - t0))

    # held-out evaluation: the probe against cosine-to-prototype
    protos = np.stack([X[tr][Ytr[:, k] > 0].mean(0) if (Ytr[:, k] > 0).any() else np.zeros(D) for k in range(K)])
    protos /= np.maximum(np.linalg.norm(protos, axis=1, keepdims=True), 1e-12)
    lt = predict(model, X[te])
    pt = 1 / (1 + np.exp(-lt))
    with np.errstate(all="ignore"):  # Accelerate's matmul raises spurious FP warnings on macOS
        cos_v, cos_t = X[va] @ protos.T, X[te] @ protos.T
    lv = predict(model, X[va])
    rows, thr_p, thr_c = [], [], []
    for k, name in enumerate(names):
        yv, yt = Y[va][:, k] > 0, Y[te][:, k] > 0
        tp, _ = metrics.best_f1(lv[:, k], yv)
        tc, _ = metrics.best_f1(cos_v[:, k], yv)
        thr_p.append(tp)
        thr_c.append(tc)
        rel, ece = metrics.reliability(pt[:, k], yt)
        rows.append({"concept": name, "description": desc.get(name, ""), "positives": int(yt.sum()), "n": int(len(yt)),
                     "auc_probe": metrics.auc(lt[:, k], yt), "auc_cosine": metrics.auc(cos_t[:, k], yt),
                     "f1_probe": metrics.f1_at(lt[:, k], yt, tp), "f1_cosine": metrics.f1_at(cos_t[:, k], yt, tc),
                     "ece_probe": ece, "reliability": rel})
    mean = lambda key: float(np.mean([r[key] for r in rows if r[key] is not None]))
    summary = {k: mean(k) for k in ("auc_probe", "auc_cosine", "f1_probe", "f1_cosine", "ece_probe")}
    run.event("probes", total - 1, rows=rows, split={"train": int(tr.sum()), "validation": int(va.sum()),
                                                     "test": int(te.sum())})
    run.log(total - 1, **summary)
    run.summary(**summary)
    W = {k: v for k, v in model.state_dict().items()}
    np.savez(os.path.join(run.dir, "probes.npz"), names=np.array(names), prototypes=protos.astype(np.float32),
             threshold_probe=np.array(thr_p, np.float32), threshold_cosine=np.array(thr_c, np.float32),
             hidden=np.array(int(run.p("hidden", 0))), **{"w/" + k: v for k, v in W.items()})
    run.event("info", total - 1, text=f"Saved the filters to {os.path.join(run.dir, 'probes.npz')}")


def try_it(run, req):
    """Score a text with every filter"""
    from rawgpu import nn
    d = np.load(os.path.join(run.dir, "probes.npz"))
    names = [str(n) for n in d["names"]]
    enc = eg2_encoder()
    x = enc.encode([req.get("text", "")], prompt="document")
    del enc
    model = build(x.shape[1], len(names), int(d["hidden"]), np.random.default_rng(0))
    model.load_state_dict({k[2:]: d[k] for k in d.files if k.startswith("w/")})
    logit = predict(model, x)[0]
    p = 1 / (1 + np.exp(-logit))
    with np.errstate(all="ignore"):
        cos = (x @ d["prototypes"].T)[0]
    nn.close()
    return {"results": [{"concept": n, "probability": float(p[i]), "cosine": float(cos[i]),
                         "fires": bool(logit[i] >= d["threshold_probe"][i]),
                         "fires_cosine": bool(cos[i] >= d["threshold_cosine"][i])} for i, n in enumerate(names)]}
