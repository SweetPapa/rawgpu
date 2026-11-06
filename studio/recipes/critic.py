"""A Jev-style critic: state -> calibrated probabilities over the options of a typed question.

Jev (TypeSafe AI) answers questions about a state with probabilities instead of text: Choice (one of a set
of options), Score (a level on an ordered scale) and Noul (yes or no). This critic does the same at small
scale. EmbeddingGemma 2 embeds the state, the question and each option's description; a two-tower head
maps (state, question) and (option, question) into one space, and the dot products, softmaxed over the
question's options, are the probabilities. Because options are text, the critic can be asked a new
question with new options; it answers best near what it was trained on.

After training, temperature scaling on the validation split calibrates the probabilities (Jev's selling
point), and the test split measures it: expected calibration error, Brier score, NLL and reliability
diagrams, before and after. Confidence is Jev's (J * p_max - 1) / (J - 1).

Data: questions generated from your code index (the language, the kind of file, its size, whether it is
a test), or a JSONL file of {"state", "question", "type", "options", "answer"} lines.
"""
import json
import os
import time

import numpy as np

from . import metrics
from .codeindex import CodeIndex, KINDS, SIZES
from .probe import eg2_encoder, split_by

META = {
    "name": "Jev-style critic",
    "group": "On EmbeddingGemma 2",
    "tagline": "State in, calibrated probabilities out: Choice, Score and yes/no",
    "description": "The fast lane's critic: ask a typed question about a state and get a probability for each "
                   "option, not text. Options are written in plain language and embedded, so new questions work "
                   "too. Temperature scaling calibrates it on held-out data, and the reliability diagram shows "
                   "whether a stated 70% is right 70% of the time.",
    "params": [
        {"key": "source", "label": "Data", "type": "choice", "default": "code index",
         "choices": ["code index", "JSONL file"]},
        {"key": "embeddings", "label": "Embeddings", "type": "choice", "default": "content only",
         "choices": ["content only", "as indexed (with the file path)"],
         "help": "The index embedded each chunk with its path as a title, and these labels come from paths: "
                 "\"content only\" re-embeds the text alone on the card (cached after the first time)."},
        {"key": "index", "label": "Code index", "type": "path", "default": "~/rawgpu-index", "advanced": True},
        {"key": "jsonl", "label": "JSONL file", "type": "path", "default": "",
         "help": 'One {"state", "question", "type": "choice"|"score"|"noul", "options": {name: description}, '
                 '"answer"} per line.'},
        {"key": "per_question", "label": "Examples per question", "type": "int", "default": 2500, "min": 200},
        {"key": "dim", "label": "Joint space", "type": "choice", "default": 256, "choices": [64, 128, 256, 512]},
        {"key": "hidden", "label": "Hidden units", "type": "choice", "default": 512, "choices": [256, 512, 1024]},
        {"key": "steps", "label": "Steps", "type": "int", "default": 1500, "min": 10},
        {"key": "lr", "label": "Learning rate", "type": "float", "default": 1e-3},
        {"key": "batch", "label": "Batch", "type": "int", "default": 512, "advanced": True},
    ],
    "tiles": [
        {"key": "test_accuracy", "label": "Test accuracy", "fmt": ".1%", "best": "max"},
        {"key": "test_ece", "label": "Calibration error", "fmt": ".3f", "best": "min",
         "help": "after temperature scaling; 0 is perfect"},
        {"key": "test_brier", "label": "Brier score", "fmt": ".3f", "best": "min"},
        {"key": "temperature", "label": "Temperature", "fmt": ".2f"},
    ],
    "charts": [
        {"title": "Loss", "series": [{"key": "loss", "label": "Training"}, {"key": "val_loss", "label": "Validation"}]},
        {"title": "Validation accuracy", "series": [{"key": "val_accuracy", "label": "Accuracy"}], "small": True,
         "fmt": ".0%"},
    ],
    "panels": ["reliability"],
}

NOUL = {"yes": "yes, it is", "no": "no, it is not"}


def code_questions(ci, n_per, rng):
    """Questions about code chunks, answers from their paths: [(question, chunk indices, answer names)]"""
    langs = ci.top(ci.lang, 10)
    qs = [
        ({"id": "language", "type": "choice", "text": "What language is this written in?",
          "paraphrases": ["Which programming language is this?", "What is the language of this code?",
                          "Identify the language this snippet uses."],
          "options": {l: f"code or text written in {l}" for l in langs}}, ci.lang),
        ({"id": "kind", "type": "choice", "text": "What kind of file is this part of?",
          "paraphrases": ["What sort of file does this come from?", "Is this code, tests, docs, config or build?",
                          "Classify the file this belongs to."], "options": KINDS}, ci.kind),
        ({"id": "size", "type": "score", "text": "How large is the file this comes from?",
          "paraphrases": ["How big is the source file?", "Estimate the size of the whole file.",
                          "Is the file this is from small or large?"],
          "options": {name: desc for name, desc, _ in SIZES}}, ci.size),
        ({"id": "is_test", "type": "noul", "text": "Is this part of a test suite?",
          "paraphrases": ["Is this test code?", "Does this belong to the tests?", "Is this a unit test?"],
          "options": NOUL}, np.where(ci.kind == "tests", "yes", "no")),
    ]
    out = []
    for q, col in qs:
        ok = np.flatnonzero(np.isin(col, list(q["options"])))
        idx = rng.choice(ok, min(n_per, len(ok)), replace=False)
        out.append((q, idx, col[idx]))
    return out


def load(run, rng):
    """Everything the training needs: state vectors, question and option tables, examples"""
    run.set_status(message="Preparing the data")
    if run.p("source") == "JSONL file":
        path = os.path.expanduser(run.p("jsonl", ""))
        rows = [json.loads(l) for l in open(path) if l.strip()]
        qmap = {}
        for r in rows:
            t = r.get("type", "choice")
            opts = r.get("options") or (NOUL if t == "noul" else {})
            if isinstance(opts, list):
                opts = {o: o for o in opts}
            q = qmap.setdefault(r["question"], {"id": f"q{len(qmap)}", "type": t, "text": r["question"], "options": {}})
            q["options"].update(opts)
        texts = [r["state"] for r in rows]
        run.set_status(message=f"Embedding {len(texts):,} states on the card")
        enc = eg2_encoder()
        S = enc.encode(texts, prompt="document")
        ex = []
        for i, r in enumerate(rows):
            a = r["answer"]
            if isinstance(a, bool):
                a = "yes" if a else "no"
            ex.append((r["question"], i, str(a)))
        questions = list(qmap.values())
        split_keys = texts
        what = f"{path}: {len(rows):,} examples, {len(questions)} questions"
    else:
        ci = CodeIndex(run.p("index"))
        qs = code_questions(ci, int(run.p("per_question")), rng)
        questions = [q for q, _, _ in qs]
        allidx = np.unique(np.concatenate([idx for _, idx, _ in qs]))
        pos = {c: i for i, c in enumerate(allidx)}
        content = run.p("embeddings", "content only") == "content only"
        S = ci.content_vectors(allidx, lambda m: run.set_status(message=m)) if content else ci.vectors(allidx)
        ex = [(q["text"], pos[c], str(a)) for q, idx, ans in qs for c, a in zip(idx, ans)]
        split_keys = [ci.path[c] for c in allidx]
        what = (f"code index: {len(ex):,} examples about {len(allidx):,} chunks "
                f"({'text only' if content else 'embedded with their paths'}), {len(questions)} questions")
        enc = eg2_encoder()
    # the question and option embeddings
    # each question in its phrasings, each option as "name: description", the name, and the description:
    # training draws one of each at random, so new phrasings and bare option names work too
    opt_names, opt_q, opt_texts = [], [], []
    for qi, q in enumerate(questions):
        for name, desc in q["options"].items():
            opt_names.append(name)
            opt_q.append(qi)
            forms = [f"{name}: {desc}" if desc and desc != name else name, name]
            if desc and desc != name:
                forms.append(desc)
            opt_texts.append(forms)
    run.set_status(message="Embedding the questions and options")
    q_texts = [[q["text"]] + list(q.get("paraphrases", [])) for q in questions]
    flat_q = enc.encode([t for ts in q_texts for t in ts], prompt="classification")
    flat_o = enc.encode([t for ts in opt_texts for t in ts], prompt="classification")
    del enc
    Qvar, Ovar, i = [], [], 0
    for ts in q_texts:
        Qvar.append(flat_q[i:i + len(ts)])
        i += len(ts)
    i = 0
    for ts in opt_texts:
        Ovar.append(flat_o[i:i + len(ts)])
        i += len(ts)
    Qv = np.stack([v[0] for v in Qvar])
    Ov = np.stack([v[0] for v in Ovar])
    qindex = {q["text"]: i for i, q in enumerate(questions)}
    opt_index = {(qi, n): j for j, (n, qi) in enumerate(zip(opt_names, opt_q))}
    E = []
    for qt, si, a in ex:
        qi = qindex[qt]
        if (qi, a) in opt_index:
            E.append((si, qi, opt_index[(qi, a)]))
    E = np.array(E, np.int64)
    split_state = split_by(split_keys, int(run.p("seed", 0)))
    return dict(S=S, Qv=Qv, Ov=Ov, Qvar=Qvar, Ovar=Ovar, opt_q=np.array(opt_q), opt_names=opt_names,
                questions=questions, E=E, split=split_state[E[:, 0]], what=what)


def build(p, rng, D=768):
    from rawgpu import nn
    d, h = int(p["dim"]), int(p["hidden"])

    class Critic(nn.Module):
        def __init__(self):
            self.state = nn.MLP([2 * D, h, d], act="gelu", norm=True, rng=rng)
            self.option = nn.MLP([2 * D, h, d], act="gelu", norm=True, rng=rng)
            self.scale = 1.0 / np.sqrt(d)

        def forward(self, s, q, o):
            """s, q: [B, D] (state, its question); o: [n_options, 2D] (option, its question) -> logits [B, n]"""
            a = self.state(nn.concat_cols([s, q]))
            b = self.option(o)
            return nn.matmul(a, b, tb=True, alpha=self.scale)
    return Critic()


def logits_for(model, data, rows, batch=4096):
    """Masked logits [len(rows), n_options] (options of other questions at -1e4)"""
    from rawgpu import nn
    S, Qv, Ov, opt_q = data["S"], data["Qv"], data["Ov"], data["opt_q"]
    O = nn.tensor(np.concatenate([Ov, Qv[opt_q]], 1))
    out = []
    with nn.no_grad():
        for i in range(0, len(rows), batch):
            e = rows[i:i + batch]
            out.append(model(nn.tensor(S[e[:, 0]]), nn.tensor(Qv[e[:, 1]]), O).numpy())
    lg = np.concatenate(out) if out else np.zeros((0, len(opt_q)), np.float32)
    return np.where(opt_q[None, :] == rows[:, 1:2], lg, -1e4)


def evaluate(lg, rows, data, T):
    """Overall and per-question metrics at temperature T, and the reliability bins"""
    p = metrics.softmax(lg, T)
    labels = rows[:, 2]
    overall, rel = metrics.multiclass(p, labels)
    per = []
    for qi, q in enumerate(data["questions"]):
        m = rows[:, 1] == qi
        if m.sum() < 10:
            continue
        cols = np.flatnonzero(data["opt_q"] == qi)
        pq = p[m][:, cols]
        pq /= pq.sum(1, keepdims=True)
        lab = np.searchsorted(cols, labels[m])
        r, rr = metrics.multiclass(pq, lab)
        J = len(cols)
        conf = ((J * pq.max(1) - 1) / (J - 1)).mean() if J > 1 else 1.0
        per.append(dict(id=q["id"], question=q["text"], type=q["type"], options=J, confidence=float(conf), reliability=rr,
                        **r))
    return overall, rel, per


def train(run):
    from rawgpu import nn
    seed = int(run.p("seed", 0))
    rng = np.random.default_rng(seed)
    data = load(run, rng)
    E, split = data["E"], data["split"]
    tr, va, te = E[split == 0], E[split == 1], E[split == 2]
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError(f"too few examples to split ({len(E)}: {len(tr)} train, {len(va)} validation, {len(te)} test); "
                         f"states are split 80/10/10, so use at least a few dozen distinct states")
    run.event("info", 0, text=f"{data['what']}; {len(tr):,} train / {len(va):,} validation / {len(te):,} test; "
                              f"{len(data['opt_names'])} options in all")
    model = build(run.params, rng)
    total, B = int(run.p("steps")), int(run.p("batch"))
    opt = nn.AdamW(model.parameters(), lr=float(run.p("lr")), weight_decay=0.01, max_grad_norm=1.0)
    S, Qv, Ov, opt_q = data["S"], data["Qv"], data["Ov"], data["opt_q"]
    Qvar, Ovar = data["Qvar"], data["Ovar"]
    run.saver = lambda: run.save("latest", model=model, step=run._step)
    run.set_status(message="")
    best, t0 = float("inf"), time.time()
    for step in run.steps(0, total):
        lr = nn.cosine_lr(step, total, float(run.p("lr")), 50)
        e = tr[rng.integers(0, len(tr), B)]
        mask = np.where(opt_q[None, :] == e[:, 1:2], 0.0, -1e4).astype(np.float32)
        Qs = np.stack([v[rng.integers(len(v))] for v in Qvar])   # this step's phrasing of each question
        Os = np.stack([v[rng.integers(len(v))] for v in Ovar])   # and of each option
        O_host = np.concatenate([Os, Qs[opt_q]], 1)
        lg = model(nn.tensor(S[e[:, 0]]), nn.tensor(Qs[e[:, 1]]), nn.tensor(O_host)) + nn.tensor(mask)
        loss = nn.cross_entropy(lg, e[:, 2])
        opt.zero_grad()
        loss.backward()
        opt.step(lr)
        if run.every(max(1, total // 60)) or step == total - 1:
            lv = logits_for(model, data, va)
            o, _, _ = evaluate(lv, va, data, 1.0)
            run.log(step, loss=loss.item(), val_loss=o["nll"], val_accuracy=o["accuracy"], val_ece=o["ece"], lr=lr,
                    examples_s=(step + 1) * B / (time.time() - t0))
            if o["nll"] < best:
                best = o["nll"]
                run.save("best", model=model, step=step)
    run.load("best", model=model)
    lv, lt = logits_for(model, data, va), logits_for(model, data, te)
    T = metrics.fit_temperature(lv, va[:, 2])
    before, rel_b, _ = evaluate(lt, te, data, 1.0)
    after, rel_a, per = evaluate(lt, te, data, T)
    run.event("reliability", total - 1, temperature=T, before=before, after=after, bins_before=rel_b, bins_after=rel_a,
              questions=per)
    summary = {"test_accuracy": after["accuracy"], "test_ece": after["ece"], "test_brier": after["brier"],
               "test_nll": after["nll"], "ece_before": before["ece"], "temperature": T}
    run.log(total - 1, **summary)
    run.summary(**summary)
    run.save("best", model=model, step=total - 1, temperature=T, Qv=Qv, Ov=Ov, opt_q=opt_q,
             opt_names=np.array(data["opt_names"]), questions=np.array(json.dumps(data["questions"])))


def try_it(run, req):
    """Ask a question about a state: one of the trained questions (question_id), or a new one (question,
    type, options {name: description})"""
    from rawgpu import nn
    model = build(run.params, np.random.default_rng(0))
    ex = run.load("best", model=model)
    questions = json.loads(str(ex["questions"]))
    T = float(ex["temperature"]) if "temperature" in ex else 1.0
    enc = eg2_encoder()
    s = enc.encode([req.get("state", "")], prompt="document")
    if req.get("question_id"):
        qi = next(i for i, q in enumerate(questions) if req["question_id"] in (q["id"], q["text"]))
        q = questions[qi]
        cols = np.flatnonzero(ex["opt_q"] == qi)
        qv = ex["Qv"][qi:qi + 1]
        O = np.concatenate([ex["Ov"][cols], np.repeat(qv, len(cols), 0)], 1)
        names = [str(ex["opt_names"][c]) for c in cols]
    else:
        q = {"text": req["question"], "type": req.get("type", "choice"), "options": req.get("options") or NOUL}
        names = list(q["options"])
        texts = [f"{n}: {d}" if d and d != n else n for n, d in q["options"].items()]
        qv = enc.encode([q["text"]], prompt="classification")
        ov = enc.encode(texts, prompt="classification")
        O = np.concatenate([ov, np.repeat(qv, len(names), 0)], 1)
    del enc
    with nn.no_grad():
        lg = model(nn.tensor(s), nn.tensor(qv), nn.tensor(O)).numpy()[0]
    p = metrics.softmax(lg[None], T)[0]
    J = len(names)
    out = {"question": q["text"], "type": q["type"], "temperature": T,
           "options": [{"name": n, "probability": float(pp)} for n, pp in zip(names, p)],
           "choice": names[int(p.argmax())], "confidence": float((J * p.max() - 1) / (J - 1)) if J > 1 else 1.0}
    if q["type"] == "score":
        out["position"] = float((p * np.arange(J)).sum())
    if q["type"] == "noul":
        out["probability_yes"] = float(p[names.index("yes")]) if "yes" in names else float(p[0])
    nn.close()
    return out
