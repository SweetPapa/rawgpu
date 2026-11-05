#!/usr/bin/env python3
"""Search the code index from another program: JSON lines on stdin, one JSON reply per request on stdout.

  local/eg2-venv/bin/python models/embeddinggemma2/search_server.py [--out ~/rawgpu-index] [--release-after 20]

Requests (each may carry an "id", echoed in the reply):
  {"op": "info"}                       the index: chunks, files, size, projects with their chunk counts
  {"op": "search", "query": "...", "backend": "rawgpu" | "mps" | "cpu", "k": 10, "task": "code",
   "projects": [...], "langs": ["Python", ...], "path": ["src/", ...],   (filters, each optional)
   "dedupe": true,   (fold identical chunks from copies of a project)
   "group": true}    (the top k files, each with its best chunks; false: the top k chunks)
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


LANGS = {  # file extension -> language, for filters and labels
    "py": "Python", "pyi": "Python", "ipynb": "Python", "c": "C", "h": "C", "cc": "C++", "cpp": "C++",
    "cxx": "C++", "hpp": "C++", "hh": "C++", "hxx": "C++", "ino": "C++", "m": "Objective-C",
    "mm": "Objective-C", "swift": "Swift", "rs": "Rust", "go": "Go", "java": "Java", "kt": "Kotlin",
    "kts": "Kotlin", "cs": "C#", "js": "JavaScript", "mjs": "JavaScript", "cjs": "JavaScript",
    "jsx": "JavaScript", "ts": "TypeScript", "tsx": "TypeScript", "mts": "TypeScript", "cts": "TypeScript",
    "vue": "Vue", "svelte": "Svelte", "html": "HTML", "htm": "HTML", "css": "CSS", "scss": "CSS",
    "less": "CSS", "md": "Markdown", "mdx": "Markdown", "markdown": "Markdown", "rst": "Text", "txt": "Text",
    "json": "JSON", "jsonc": "JSON", "yaml": "YAML", "yml": "YAML", "toml": "TOML", "ini": "INI", "cfg": "INI",
    "sh": "Shell", "bash": "Shell", "zsh": "Shell", "fish": "Shell", "ps1": "PowerShell", "sql": "SQL",
    "tf": "Terraform", "tfvars": "Terraform", "hcl": "Terraform", "gd": "GDScript", "dart": "Dart",
    "lua": "Lua", "rb": "Ruby", "php": "PHP", "r": "R", "cl": "OpenCL", "metal": "Metal", "glsl": "Shader",
    "hlsl": "Shader", "wgsl": "Shader", "shader": "Shader", "cmake": "CMake", "mk": "Make",
    "proto": "Protobuf", "graphql": "GraphQL", "gql": "GraphQL", "xml": "XML", "plist": "XML", "csv": "CSV",
    "xsd": "XML", "storyboard": "XML", "xib": "XML", "pkgproj": "XML", "xojo_code": "Xojo", "xojo_window": "Xojo",
    "xojo_project": "Xojo", "xojo_script": "Xojo", "rbbas": "Xojo", "po": "Gettext", "pot": "Gettext",
    "swiftinterface": "Swift", "tex": "LaTeX", "sty": "LaTeX", "bst": "LaTeX", "sol": "Solidity", "move": "Move",
    "gradle": "Gradle", "bat": "Batch", "cmd": "Batch", "vbs": "VBScript", "nsh": "NSIS", "nsi": "NSIS",
    "ld": "Linker script", "pbxproj": "Xcode project", "fountain": "Fountain",
}
NAMED = {"makefile": "Make", "gnumakefile": "Make", "dockerfile": "Dockerfile", "cmakelists.txt": "CMake",
         "gemfile": "Ruby", "rakefile": "Ruby"}


def lang_of(path):
    base = path.rsplit("/", 1)[-1].lower()
    if base in NAMED:
        return NAMED[base]
    if base.startswith("dockerfile"):
        return "Dockerfile"
    return LANGS.get(base.rsplit(".", 1)[-1] if "." in base else "", "Other")


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
        rows = self.db.execute("SELECT project, path FROM chunks ORDER BY id").fetchall()
        assert len(rows) == len(self.ids)
        self.projects = sorted({p for p, _ in rows})
        pcode = {p: i for i, p in enumerate(self.projects)}
        self.files = sorted({f for _, f in rows})
        fcode = {f: i for i, f in enumerate(self.files)}
        self.files_lower = [f.lower() for f in self.files]
        self.file_of = np.fromiter((fcode[f] for _, f in rows), np.int32, len(rows))  # chunk -> file
        self.proj = np.fromiter((pcode[p] for p, _ in rows), np.int32, len(rows))     # chunk -> project
        self.file_proj = np.zeros(len(self.files), np.int32)
        self.file_proj[self.file_of] = self.proj
        file_lang = [lang_of(f) for f in self.files]
        self.langs = sorted(set(file_lang))
        lcode = {name: i for i, name in enumerate(self.langs)}
        self.file_lang = np.array([lcode[x] for x in file_lang], np.int32)
        self.counts = np.bincount(self.proj, minlength=len(self.projects))
        self.lang_counts = np.bincount(self.file_lang[self.file_of], minlength=len(self.langs))
        self.file_chunks = np.bincount(self.file_of, minlength=len(self.files))
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
        order = np.argsort(-self.lang_counts)
        return {"chunks": int(len(self.ids)), "files": len(self.files), "bytes": os.path.getsize(self.path),
                "root": self.root, "built": self.mtime,
                "projects": [[p, int(c)] for p, c in zip(self.projects, self.counts)],
                "languages": [[self.langs[i], int(self.lang_counts[i])] for i in order]}

    def file_mask(self, projects=None, langs=None, path=None):
        """Which files pass the filters (None: all of them)."""
        if not (projects or langs or path):
            return None
        m = np.ones(len(self.files), bool)
        if projects:
            codes = [self.projects.index(p) for p in projects if p in self.projects]
            m &= np.isin(self.file_proj, codes)
        if langs:
            codes = [self.langs.index(x) for x in langs if x in self.langs]
            m &= np.isin(self.file_lang, codes)
        if path:
            needles = [p.lower() for p in path]
            m &= np.fromiter((all(n in f for n in needles) for f in self.files_lower), bool, len(self.files))
        return m

    def search(self, qv, k, projects=None, langs=None, path=None, dedupe=True, group=True):
        """The top k chunks, or with group, the top k files, each with its best-matching chunks."""
        with np.errstate(all="ignore"):  # Accelerate's sgemv raises spurious FP flags
            s = np.asarray(self.V @ qv)
        fm = self.file_mask(projects, langs, path)
        if fm is not None:
            s = np.where(fm[self.file_of], s, -2.0)
        n = min(len(s), max(k * 30, 600) if group else max(k * 8, 64))
        cand = np.argpartition(-s, n - 1)[:n]
        cand = cand[np.argsort(-s[cand])]
        hits, seen, groups = [], {}, {}
        for j in cand:
            if s[j] <= -2.0:
                break
            fi = int(self.file_of[j])
            if group and len(groups) >= k and fi not in groups:
                continue  # enough files: only gather more matches within them
            a, b, text, tokens = self.db.execute(
                "SELECT start_line, end_line, text, tokens FROM chunks WHERE id = ?", (int(self.ids[j]),)).fetchone()
            path_j = self.files[fi]
            key = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
            if dedupe and key in seen:  # the same text in another copy of a project
                seen[key]["copies"].append(f"{path_j}:{a}")
                continue
            hit = {"score": float(s[j]), "start": a, "end": b, "tokens": tokens, "text": text, "copies": []}
            seen[key] = hit
            if not group:
                hit.update(path=path_j, abs_path=os.path.join(self.root, path_j),
                           project=self.projects[self.proj[j]], lang=self.langs[self.file_lang[fi]])
                hits.append(hit)
                if len(hits) >= k:
                    break
                continue
            g = groups.get(fi)
            if g is None:
                g = groups[fi] = {"path": path_j, "abs_path": os.path.join(self.root, path_j),
                                  "project": self.projects[self.proj[j]], "lang": self.langs[self.file_lang[fi]],
                                  "score": hit["score"], "file_chunks": int(self.file_chunks[fi]), "hits": []}
            if len(g["hits"]) < 8:
                g["hits"].append(hit)
        if not group:
            return hits
        out = list(groups.values())  # already in order of each file's best chunk
        for g in out:
            g["copies"] = sorted({c for h in g["hits"] for c in h["copies"]})
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
                    as_list = lambda v: [v] if isinstance(v, str) else (v or None)  # noqa: E731
                    group = bool(req.get("group", True))
                    res = index.search(np.asarray(qv, np.float32), int(req.get("k", 10)),
                                       projects=as_list(req.get("projects") or req.get("project")),
                                       langs=as_list(req.get("langs")), path=as_list(req.get("path")),
                                       dedupe=bool(req.get("dedupe", True)), group=group)
                    t2 = time.perf_counter()
                    send({"id": rid, "ok": True, ("groups" if group else "results"): res, "backend": backend,
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
