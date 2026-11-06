"""A small GPT trained from scratch on bytes: any text file or folder, by default rawgpu's own source."""
import glob
import os
import subprocess
import time

import numpy as np

from run import ROOT

META = {
    "name": "Tiny GPT",
    "group": "From scratch",
    "tagline": "A byte-level language model trained on text you choose",
    "description": "A decoder-only transformer that reads bytes, so any text works without a tokenizer. "
                   "By default it learns rawgpu's own source and docs (3.4 MB). Watch the validation loss "
                   "fall and the samples go from noise to plausible C and Markdown.",
    "params": [
        {"key": "corpus", "label": "Text", "type": "path", "default": "",
         "help": "A text file, a folder (its text files, recursively), or a glob. Empty: this repo's source."},
        {"key": "dim", "label": "Width", "type": "choice", "default": 256, "choices": [128, 192, 256, 384, 512]},
        {"key": "layers", "label": "Layers", "type": "int", "default": 6, "min": 1, "max": 24},
        {"key": "heads", "label": "Heads", "type": "int", "default": 8, "min": 1, "max": 16},
        {"key": "seq_len", "label": "Context", "type": "choice", "default": 256, "choices": [64, 128, 256, 512],
         "unit": "bytes"},
        {"key": "batch", "label": "Batch", "type": "int", "default": 32, "min": 1, "max": 256},
        {"key": "steps", "label": "Steps", "type": "int", "default": 2000, "min": 10,
         "help": "On the default 3.4 MB corpus, validation loss bottoms out near 2000 (later steps overfit)."},
        {"key": "lr", "label": "Learning rate", "type": "float", "default": 1e-3},
        {"key": "warmup", "label": "Warmup steps", "type": "int", "default": 100, "advanced": True},
        {"key": "weight_decay", "label": "Weight decay", "type": "float", "default": 0.1, "advanced": True},
        {"key": "eval_every", "label": "Evaluate every", "type": "int", "default": 250, "unit": "steps",
         "advanced": True},
        {"key": "sample_every", "label": "Sample every", "type": "int", "default": 500, "unit": "steps",
         "advanced": True},
        {"key": "prompt", "label": "Sample prompt", "type": "str", "default": "/* ", "advanced": True},
    ],
    "tiles": [
        {"key": "val_loss", "label": "Validation loss", "fmt": ".3f", "best": "min", "unit": "nats/byte"},
        {"key": "val_bpb", "label": "Bits per byte", "fmt": ".3f", "best": "min"},
        {"key": "tok_s", "label": "Throughput", "fmt": ",.0f", "unit": "bytes/s"},
    ],
    "charts": [
        {"title": "Loss", "series": [{"key": "loss", "label": "Training"}, {"key": "val_loss", "label": "Validation"}],
         "unit": "nats/byte"},
        {"title": "Learning rate", "series": [{"key": "lr", "label": "Learning rate"}], "small": True},
        {"title": "Gradient norm", "series": [{"key": "grad_norm", "label": "Gradient norm"}], "small": True},
    ],
    "panels": ["samples"],
}

TEXT_EXT = {".md", ".txt", ".c", ".h", ".py", ".cl", ".rs", ".js", ".ts", ".css", ".html", ".sh", ".json", ".toml",
            ".yml", ".yaml", ".swift", ".go", ".java", ".cpp", ".hpp", ".m"}


def load_corpus(spec):
    """Bytes of the corpus and a description of where they came from"""
    if not spec:
        files = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True).stdout.split()
        files = [os.path.join(ROOT, f) for f in files if os.path.splitext(f)[1] in TEXT_EXT and "vendor/" not in f]
        what = "rawgpu's source and docs"
    else:
        spec = os.path.expanduser(spec)
        if os.path.isdir(spec):
            files = [p for p in glob.glob(os.path.join(spec, "**", "*"), recursive=True)
                     if os.path.isfile(p) and os.path.splitext(p)[1].lower() in TEXT_EXT
                     and "/node_modules/" not in p and "/.git/" not in p]
        else:
            files = sorted(glob.glob(spec)) or [spec]
        what = spec
    parts = []
    for f in sorted(files):
        try:
            with open(f, "rb") as fh:
                b = fh.read(4 << 20)
            if b"\0" not in b[:4096]:
                parts.append(b)
        except OSError:
            pass
    data = b"\n\n".join(parts)
    if len(data) < 10000:
        raise ValueError(f"the corpus is too small ({len(data)} bytes from {len(parts)} files): {what}")
    return np.frombuffer(data, np.uint8), f"{what}: {len(parts)} files, {len(data) / 1e6:.1f} MB"


def build(p):
    from rawgpu import nn
    return nn.GPT(256, dim=int(p["dim"]), heads=int(p["heads"]), layers=int(p["layers"]), seq_len=int(p["seq_len"]),
                  seed=int(p.get("seed", 0)))


def train(run):
    from rawgpu import nn
    p = run.params
    data, what = load_corpus(run.p("corpus"))
    n_val = max(len(data) // 20, int(p["seq_len"]) * 64)
    train_d, val_d = data[:-n_val], data[-n_val:]
    rng = np.random.default_rng(int(p.get("seed", 0)))
    model = build(p)
    B, T, total = int(p["batch"]), int(p["seq_len"]), int(p["steps"])
    opt = nn.AdamW(model.parameters(), lr=float(p["lr"]), betas=(0.9, 0.95), weight_decay=float(p["weight_decay"]),
                   max_grad_norm=1.0)
    start = 0
    if run.resume:
        extra = run.load(model=model, opt=opt)
        if extra:
            start = int(extra["step"]) + 1
    run.saver = lambda: run.save(model=model, opt=opt, step=run._step)
    run.event("info", start, text=f"{model.num_parameters() / 1e6:.2f}M parameters; corpus {what}; "
                                  f"{B * T:,} bytes per step")
    run.summary(params=model.num_parameters(), corpus=what)

    def batch(d, n):
        idx = rng.integers(0, len(d) - T - 1, n)
        x = np.stack([d[i:i + T + 1] for i in idx]).astype(np.int32)
        return x[:, :-1], x[:, 1:]

    vx, vy = batch(val_d, 8 * B)  # a fixed validation set

    def evaluate():
        tot = 0.0
        with nn.no_grad():
            for i in range(0, len(vx), B):
                tot += nn.cross_entropy(model(vx[i:i + B]), vy[i:i + B].reshape(-1)).item() * len(vx[i:i + B])
        return tot / len(vx)

    acc, n_acc, t_log, best = None, 0, time.time(), float("inf")
    log_every = max(1, min(25, total // 100))
    for step in run.steps(start, total):
        lr = nn.cosine_lr(step, total, float(p["lr"]), int(p["warmup"]))
        x, y = batch(train_d, B)
        loss = nn.cross_entropy(model(x), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step(lr)
        with nn.no_grad():
            acc = loss.detach() if acc is None else acc + loss.detach()
        n_acc += 1
        last = step == total - 1
        if run.every(log_every) or last:
            l = acc.item() / n_acc
            dt = time.time() - t_log
            run.log(step, loss=l, lr=lr, grad_norm=opt.grad_norm(), tok_s=n_acc * B * T / dt)
            acc, n_acc, t_log = None, 0, time.time()
        if run.every(int(p["eval_every"])) or last:
            v = evaluate()
            best = min(best, v)
            run.log(step, val_loss=v, val_bpb=v / np.log(2))
            run.summary(best_val_loss=best, val_bpb=v / np.log(2))
            if v <= best:
                run.save("best", model=model, step=step)
            t_log = time.time()
        if run.every(int(p["sample_every"])) or last:
            prompt = (run.p("prompt", "") or "\n").encode()
            ids = model.generate(list(prompt), 320, temperature=0.8, top_k=40, rng=rng)
            run.event("sample", step, prompt=prompt.decode(errors="replace"),
                      text=bytes(ids[len(prompt):]).decode(errors="replace"))
            t_log = time.time()
        if run.every(1000):
            run.save(model=model, opt=opt, step=step)
            t_log = time.time()
    run.save(model=model, opt=opt, step=total - 1)


def try_it(run, req):
    """Continue a prompt with the best checkpoint"""
    from rawgpu import nn
    model = build(run.params)
    name = "best" if os.path.exists(run.ckpt("best")) else "latest"
    run.load(name, model=model)
    prompt = (req.get("prompt") or "\n").encode()
    n = int(min(max(int(req.get("length", 400)), 16), 2000))
    ids = model.generate(list(prompt), n, temperature=float(req.get("temperature", 0.8)), top_k=40)
    nn.close()
    return {"text": bytes(ids[len(prompt):]).decode(errors="replace"), "checkpoint": name}
