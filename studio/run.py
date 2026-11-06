"""What a recipe sees while it trains: its parameters, a place to log, checkpoints, and the card.

    def train(run):
        model = ...
        for step in run.steps(start, total):         # stops cleanly on Stop, pauses when the card runs hot
            ...
            if run.every(50): run.log(step, loss=loss.item(), lr=lr)
            if run.every(500): run.save(model=model, opt=opt, step=step)

A run is a directory: config.json (the recipe and its parameters), status.json (state, progress, the
card's readings), metrics.jsonl (one line per log), events.jsonl (samples, evaluations, charts), log.txt
(the recipe's output) and checkpoints/. The Studio's server reads them; nothing here needs the server.
"""
import json
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "models", "embeddinggemma2"))


def data_dir():
    """Where rawgpu keeps its data: $RAWGPU_HOME, this checkout's local/, or Application Support"""
    if os.environ.get("RAWGPU_HOME"):
        return os.environ["RAWGPU_HOME"]
    if os.path.exists(os.path.join(ROOT, "src", "rawgpu_lib.c")):
        return os.path.join(ROOT, "local")
    return os.path.expanduser("~/Library/Application Support/rawgpu")


def runs_dir():
    d = os.path.join(data_dir(), "runs")
    os.makedirs(d, exist_ok=True)
    return d


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


class Stop(Exception):
    """Raised inside steps() when the run is asked to stop; the runner saves and exits."""


class Run:
    def __init__(self, path, resume=False):
        self.dir = path
        self.cfg = read_json(os.path.join(path, "config.json"))
        self.params = self.cfg["params"]
        self.resume = resume
        self.status = read_json(os.path.join(path, "status.json"), {}) or {}
        self._metrics = open(os.path.join(path, "metrics.jsonl"), "a", buffering=1)
        self._events = open(os.path.join(path, "events.jsonl"), "a", buffering=1)
        self._stop = False
        self._step = 0
        self._last_status = 0.0
        self._last_card = 0.0
        self._t0 = time.time()
        self._step_t0 = None
        self.card = {}
        self.hot_c = float(self.params.get("max_temp_c", 85))
        os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
        signal.signal(signal.SIGTERM, self._on_term)
        signal.signal(signal.SIGINT, self._on_term)

    def _on_term(self, *_):
        self._stop = True

    # ---- parameters
    def p(self, key, default=None):
        v = self.params.get(key, default)
        return default if v in (None, "") else v

    # ---- progress
    def steps(self, start, total):
        """Yield step numbers start..total-1; raises Stop when asked to stop. Keeps status.json current and
        pauses while the card is at max_temp_c or hotter."""
        self.total = total
        self._step_t0 = (time.time(), start)
        for step in range(start, total):
            self._step = step
            if self._stop:
                raise Stop()
            now = time.time()
            if now - self._last_card > 5:
                self._read_card()
                while self.card.get("temperature_c", 0) >= self.hot_c and not self._stop:
                    self.set_status(message=f"Paused: the card is at {self.card['temperature_c']} °C "
                                            f"(limit {self.hot_c:.0f} °C); cooling down")
                    time.sleep(5)
                    self._read_card()
                    if self.card.get("temperature_c", 0) < self.hot_c - 8:
                        self.set_status(message="")
                        break
            if now - self._last_status > 1:
                self._write_progress()
            yield step
        self._step = total
        self._write_progress()

    def every(self, n, step=None):
        step = self._step if step is None else step
        return n and (step + 1) % int(n) == 0

    def _read_card(self):
        self._last_card = time.time()
        try:
            from rawgpu import nn
            c = nn.ctx()
            info = c.dev.info
            self.card = dict(temperature_c=info["temperature_c"], engine_mhz=info["engine_mhz"],
                             memory_mhz=info["memory_mhz"], fan_percent=info["fan_percent"],
                             vram_bytes=info["vram_bytes"], vram_free_bytes=info["vram_free_bytes"],
                             tensors_bytes=c.pool.in_use, reserved_bytes=c.pool.reserved(), at=time.time())
        except Exception as e:  # the card's readings are a nicety; never fail a run on them
            self.card = dict(error=str(e), at=time.time())

    def _write_progress(self):
        self._last_status = time.time()
        rate = None
        if self._step_t0:
            t, s = self._step_t0
            if self._step > s and time.time() > t:
                rate = (self._step - s) / (time.time() - t)
        self.set_status(step=self._step, total=getattr(self, "total", None), steps_per_s=rate, card=self.card)

    def set_status(self, **kw):
        self.status.update(kw)
        self.status["updated"] = time.time()
        write_json(os.path.join(self.dir, "status.json"), self.status)

    # ---- records
    def log(self, step, **metrics):
        """One line of metrics.jsonl: {"step", "t" (seconds since the start), and the metrics}"""
        rec = {"step": int(step), "t": round(time.time() - self._t0, 3)}
        rec.update({k: (float(v) if isinstance(v, (int, float)) or hasattr(v, "item") else v) for k, v in metrics.items()})
        self._metrics.write(json.dumps(rec) + "\n")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)

    def event(self, kind, step, **data):
        """A sample, an evaluation, a chart: the UI shows the latest of each kind (and keeps the history)"""
        rec = {"kind": kind, "step": int(step), "t": round(time.time() - self._t0, 3)}
        rec.update(data)
        self._events.write(json.dumps(rec) + "\n")

    def summary(self, **kw):
        """Headline results for the run list (best validation loss, accuracy...)"""
        s = self.status.get("summary", {})
        s.update(kw)
        self.set_status(summary=s)

    # ---- checkpoints
    def ckpt(self, name="latest"):
        return os.path.join(self.dir, "checkpoints", name + ".npz")

    def save(self, name="latest", model=None, opt=None, **extra):
        """Model weights, optimizer state and anything else (numbers, arrays) in checkpoints/<name>.npz"""
        import numpy as np
        d = {}
        if model is not None:
            d.update({"model/" + k: v for k, v in model.state_dict().items()})
        if opt is not None:
            st = opt.state()
            d["opt/t"] = np.array(st["t"])
            for i, (m, v) in enumerate(zip(st["m"], st["v"])):
                d[f"opt/m{i}"], d[f"opt/v{i}"] = m, v
        for k, v in extra.items():
            d["extra/" + k] = np.asarray(v)
        tmp = self.ckpt(name) + ".tmp.npz"
        np.savez(tmp, **d)
        os.replace(tmp, self.ckpt(name))

    def load(self, name="latest", model=None, opt=None):
        """Restore what save() wrote; returns the extras as a dict (or None without a checkpoint)"""
        import numpy as np
        path = self.ckpt(name)
        if not os.path.exists(path):
            return None
        with np.load(path) as d:
            if model is not None:
                model.load_state_dict({k[6:]: d[k] for k in d.files if k.startswith("model/")})
            if opt is not None and "opt/t" in d.files:
                n = len(opt.m)
                opt.load_state({"t": int(d["opt/t"]), "m": [d[f"opt/m{i}"] for i in range(n)],
                                "v": [d[f"opt/v{i}"] for i in range(n)]})
            return {k[6:]: d[k] for k in d.files if k.startswith("extra/")}
