#!/usr/bin/env python3
"""Search the code index from another program: JSON lines on stdin, one JSON reply per request on stdout.

  local/eg2-venv/bin/python models/embeddinggemma2/search_server.py [--out ~/rawgpu-index] [--release-after 20]

Requests (each may carry an "id", echoed in the reply):
  {"op": "info"}                       the index: chunks, files, size, projects with their chunk counts
  {"op": "search", "query": "...", "backend": "rawgpu" | "mps" | "cpu", "k": 10,
   "project": "rawgpu" | null, "dedupe": true, "task": "code"}
  {"op": "warm", "backend": "mps"}     load a Mac backend's model now (the RX 590 is only opened by a search)
  {"op": "release"}                    close the RX 590 now
  {"op": "quit"}                       close everything and exit (so does the end of stdin)
Progress notes go out as {"event": "status", "text": "..."} lines while a request runs.

Backends embed the query with the same model, EmbeddingGemma 2 in float32: "rawgpu" on the RX 590 (eg2.py),
"mps" and "cpu" with sentence-transformers on the Mac. Their vectors agree to cosine 1.000000 (compare.py),
so all three search the same index. The similarity itself is one matrix-vector product on the Mac
(Accelerate) over the memory-mapped vectors.

The RX 590 is shared: TinyGPU takes one program at a time. So this opens it only for a query, checks first
with `rawgpu who` that nothing else has it, and closes it after --release-after seconds without a query
(so `rawgpu eject` and the indexer can have it). It never exits in the middle of a GPU call: a stop request
or the end of stdin is handled between requests.
"""
import argparse
import hashlib
import json
import os
import select
import signal
import sqlite3
import subprocess
import sys
import threading
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import index_code  # noqa: E402
from eg2 import PROMPTS  # noqa: E402

RAWGPU = os.path.join(REPO, "build", "rawgpu")


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def status(text):
    send({"event": "status", "text": text})


class Index:
    def __init__(self, out):
        self.out = out
        self.path = os.path.join(out, "index.sqlite")
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.load()

    def load(self):
        self.mtime = os.path.getmtime(self.path)
        status("Loading the index vectors ...")
        self.V, self.ids = index_code.vectors(self.out, self.db)
        rows = self.db.execute("SELECT id, project FROM chunks ORDER BY id").fetchall()
        names = sorted({p for _, p in rows})
        code = {p: i for i, p in enumerate(names)}
        self.projects = names
        self.proj = np.fromiter((code[p] for _, p in rows), np.int32, len(rows))
        assert len(self.proj) == len(self.ids)
        self.counts = np.bincount(self.proj, minlength=len(names))
        threading.Thread(target=self.warm, args=(self.V,), daemon=True).start()
        root = None
        try:
            root = json.load(open(os.path.join(self.out, "config.json"))).get("root")
        except (OSError, ValueError):
            pass
        self.root = root or os.path.expanduser("~/code")

    @staticmethod
    def warm(V):
        """Read the memory-mapped vectors once, so the first search does not wait on the disk."""
        step = 1 << 16
        for i in range(0, len(V), step):
            float(V[i:i + step, 0].sum() + V[i:i + step, -1].sum())  # touches every page of the slice

    def fresh(self):
        if os.path.getmtime(self.path) != self.mtime:
            self.load()

    def info(self):
        n, files = self.db.execute("SELECT COUNT(*), COUNT(DISTINCT path) FROM chunks").fetchone()
        return {"chunks": n, "files": files, "bytes": os.path.getsize(self.path), "root": self.root,
                "projects": [[p, int(c)] for p, c in zip(self.projects, self.counts)]}

    def search(self, qv, k, project=None, dedupe=True):
        with np.errstate(all="ignore"):  # Accelerate's sgemv raises spurious FP flags
            s = np.asarray(self.V @ qv)
        if project:
            if project not in self.projects:
                return []
            s = np.where(self.proj == self.projects.index(project), s, -2.0)
        n = min(len(s), max(k * 8, 64))
        cand = np.argpartition(-s, n - 1)[:n]
        cand = cand[np.argsort(-s[cand])]
        out, seen = [], {}
        for j in cand:
            if s[j] <= -2.0:
                break
            path, a, b, text, tokens = self.db.execute(
                "SELECT path, start_line, end_line, text, tokens FROM chunks WHERE id = ?", (int(self.ids[j]),)).fetchone()
            key = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
            if dedupe and key in seen:  # the same text in another copy of a project
                seen[key]["copies"].append(f"{path}:{a}")
                continue
            r = {"score": float(s[j]), "path": path, "abs_path": os.path.join(self.root, path),
                 "project": self.projects[self.proj[j]], "start": a, "end": b, "tokens": tokens, "text": text,
                 "copies": []}
            seen[key] = r
            if len(out) < k:
                out.append(r)
            elif not dedupe:
                break
        return out


class Backends:
    def __init__(self, release_after):
        self.release_after = release_after
        self.rx = None       # EmbeddingGemma2 on the card
        self.st = {}         # sentence-transformers models by device
        self.last_rx = 0.0

    def card_user(self):
        """The program holding the card, from `rawgpu who` (which never connects), or None."""
        try:
            r = subprocess.run([RAWGPU, "who", "--json"], capture_output=True, text=True, timeout=10)
            w = json.loads(r.stdout)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None
        if w.get("client_count", 0) > 0:
            c = w.get("clients") or [{"pid": "?", "command": "another program"}]
            return f"pid {c[0]['pid']}: {' '.join(c[0]['command'].split())[:120]}"
        return None

    def embed(self, q, backend, task):
        if backend == "rawgpu":
            if self.rx is None:
                user = self.card_user()
                if user:
                    raise RuntimeError(f"The RX 590 is in use by {user}. Pick a Mac backend, or wait for it.")
                status("Loading EmbeddingGemma 2 onto the RX 590 ...")
                from eg2 import EmbeddingGemma2
                self.rx = EmbeddingGemma2(verbose=False)
            v = self.rx.encode([q], prompt=task)[0]
            self.last_rx = time.monotonic()
            return v
        if backend not in ("mps", "cpu"):
            raise ValueError(f"unknown backend {backend!r}")
        if backend not in self.st:
            status(f"Loading EmbeddingGemma 2 with PyTorch on the Mac's {'GPU (MPS)' if backend == 'mps' else 'CPU'} ...")
            import warnings
            warnings.filterwarnings("ignore")
            import torch
            from sentence_transformers import SentenceTransformer
            self.st[backend] = SentenceTransformer("google/embeddinggemma-2", device=backend,
                                                   model_kwargs={"torch_dtype": torch.float32})
        return self.st[backend].encode([q], prompt=PROMPTS[task], normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=False)[0]

    def release(self):
        if self.rx is not None:
            gpu = self.rx.gpu
            self.rx = None
            gpu.close()
            send({"event": "card", "held": False})
            return True
        return False

    def idle_timeout(self):
        if self.rx is None:
            return None
        return max(0.0, self.last_rx + self.release_after - time.monotonic())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.expanduser("~/rawgpu-index"))
    ap.add_argument("--release-after", type=float, default=20, help="close the RX 590 after this many idle seconds")
    args = ap.parse_args()

    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))  # handled between requests, never mid-GPU-call
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))

    index = Index(os.path.realpath(args.out))
    be = Backends(args.release_after)
    send({"event": "ready", "index": index.info()})
    buf = b""
    while not stop:
        try:
            ready, _, _ = select.select([0], [], [], be.idle_timeout())
        except InterruptedError:
            continue
        if not ready:
            be.release()
            continue
        chunk = os.read(0, 65536)
        if not chunk:
            break  # stdin closed: the app is gone
        buf += chunk
        while b"\n" in buf and not stop:
            line, buf = buf.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                req = json.loads(line)
            except ValueError as e:
                send({"error": f"bad request: {e}"})
                continue
            rid, op = req.get("id"), req.get("op")
            try:
                if op == "info":
                    index.fresh()
                    send({"id": rid, "ok": True, "index": index.info()})
                elif op == "search":
                    index.fresh()
                    backend = req.get("backend", "rawgpu")
                    t0 = time.perf_counter()
                    qv = be.embed(req["query"], backend, req.get("task", "code"))
                    t1 = time.perf_counter()
                    res = index.search(np.asarray(qv, np.float32), int(req.get("k", 10)), req.get("project") or None,
                                       bool(req.get("dedupe", True)))
                    t2 = time.perf_counter()
                    send({"id": rid, "ok": True, "results": res, "backend": backend,
                          "embed_ms": round((t1 - t0) * 1e3, 1), "search_ms": round((t2 - t1) * 1e3, 1),
                          "card_held": be.rx is not None})
                elif op == "warm":  # load a Mac backend ahead of the first query (never the card: it is shared)
                    backend = req.get("backend")
                    if backend in ("mps", "cpu"):
                        be.embed("warm up", backend, "code")
                    send({"id": rid, "ok": True})
                elif op == "release":
                    send({"id": rid, "ok": True, "released": be.release()})
                elif op == "quit":
                    stop.append(1)
                    send({"id": rid, "ok": True})
                else:
                    send({"id": rid, "ok": False, "error": f"unknown op {op!r}"})
            except Exception as e:  # noqa: BLE001 -- report it to the app and keep serving
                send({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"})
    be.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
