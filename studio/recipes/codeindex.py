"""Your code index (models/embeddinggemma2/index_code.py) as training data: every chunk's EmbeddingGemma 2
vector, its file, project and language, and labels derived from them. Read-only; nothing leaves the Mac."""
import os
import re
import sqlite3

import numpy as np

DEFAULT = "~/rawgpu-index"

KIND_RULES = [  # (kind, test on the lower-cased path), first match wins
    ("tests", lambda p: bool(re.search(r"(^|/)(tests?|__tests__|spec|specs)(/|$)|(^|/)test_|_test\.|\.test\.|\.spec\.", p))),
    ("documentation", lambda p: p.endswith((".md", ".mdx", ".rst", ".txt")) or "/docs/" in p or "/doc/" in p),
    ("build or CI", lambda p: bool(re.search(r"(^|/)(makefile|cmakelists\.txt|dockerfile[^/]*|build\.gradle|"
                                               r"\.github/workflows/[^/]+|[^/]+\.(mk|cmake|nsi|bat))$", p))),
    ("configuration", lambda p: p.endswith((".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".plist",
                                            ".xml", ".env.example", ".properties"))),
]
KINDS = {
    "source code": "the program itself: functions, classes, modules",
    "tests": "unit or integration tests that check the program",
    "documentation": "prose for people: READMEs, guides, notes",
    "configuration": "settings and data files: JSON, YAML, TOML, plists",
    "build or CI": "build scripts, makefiles, Dockerfiles, CI workflows",
}
SIZES = [("tiny", "under 2 KB", 2 << 10), ("small", "2 to 10 KB", 10 << 10), ("medium", "10 to 50 KB", 50 << 10),
         ("large", "over 50 KB", None)]


def kind_of(path):
    p = path.lower()
    for k, rule in KIND_RULES:
        if rule(p):
            return k
    return "source code"


def size_of(nbytes):
    for name, _, top in SIZES:
        if top is None or nbytes < top:
            return name
    return SIZES[-1][0]


class CodeIndex:
    def __init__(self, path=None):
        self.dir = os.path.expanduser(path or DEFAULT)
        db_path = os.path.join(self.dir, "index.sqlite")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"no code index at {self.dir}: build one with models/embeddinggemma2/"
                                    f"index_code.py, or give a JSONL dataset instead")
        self.db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        import index_code
        from search_server import lang_of
        self.V, self.ids = index_code.vectors(self.dir, self.db)
        rows = self.db.execute("SELECT c.path, c.project, c.start_line, c.end_line, c.tokens, f.size FROM chunks c "
                               "LEFT JOIN files f ON f.path = c.path ORDER BY c.id").fetchall()
        self.path = [r[0] for r in rows]
        self.project = np.array([r[1] for r in rows], object)
        self.tokens = np.array([r[4] for r in rows], np.int32)
        self.file_bytes = np.array([r[5] or 0 for r in rows], np.int64)
        self.lang = np.array([lang_of(p) for p in self.path], object)
        self.kind = np.array([kind_of(p) for p in self.path], object)
        self.size = np.array([size_of(b) for b in self.file_bytes], object)
        self.start = np.array([r[2] for r in rows], np.int32)

    def __len__(self):
        return len(self.ids)

    def text(self, i):
        r = self.db.execute("SELECT text FROM chunks WHERE id = ?", (int(self.ids[i]),)).fetchone()
        return r[0] if r else ""

    def texts(self, idx):
        return [self.text(i) for i in idx]

    def vectors(self, idx):
        """The index's own vectors. The index embedded "title: <project/path> | text: <chunk>", so these
        carry the file path: anything derived from the path can be read off them."""
        return np.asarray(self.V[np.asarray(idx)], np.float32)

    def content_vectors(self, idx, status=None, max_tokens=256):
        """Vectors of the chunks' text alone (their first max_tokens tokens, no path), embedded on the card
        and cached in <rawgpu data>/studio-cache.sqlite, so labels from paths can't leak into them"""
        import hashlib
        from run import data_dir
        texts = self.texts(idx)
        keys = [hashlib.sha256(f"content{max_tokens}:{t}".encode()).hexdigest() for t in texts]
        db = sqlite3.connect(os.path.join(data_dir(), "studio-cache.sqlite"))
        db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, v BLOB NOT NULL)")
        found = {}
        for i in range(0, len(keys), 900):
            part = keys[i:i + 900]
            for k, v in db.execute(f"SELECT key, v FROM vectors WHERE key IN ({','.join('?' * len(part))})", part):
                found[k] = np.frombuffer(v, np.float32)
        missing = [i for i, k in enumerate(keys) if k not in found]
        if missing:
            from rawgpu import nn
            from eg2 import EmbeddingGemma2
            if status:
                status(f"Embedding the chunks' text on the card (no paths): 0 of {len(missing):,}; cached for next time")
            enc = EmbeddingGemma2(gpu=nn.ctx().dev, max_tokens=8192, max_seq=max_tokens, verbose=False)
            for b in range(0, len(missing), 512):
                part = missing[b:b + 512]
                V = enc.encode([texts[i] for i in part], prompt="document")
                with db:
                    db.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                                   [(keys[i], V[j].tobytes()) for j, i in enumerate(part)])
                for j, i in enumerate(part):
                    found[keys[i]] = V[j]
                if status:
                    status(f"Embedding the chunks' text on the card (no paths): {b + len(part):,} of {len(missing):,}; "
                           f"cached for next time")
            del enc
        db.close()
        return np.stack([found[k] for k in keys]).astype(np.float32)

    def top(self, values, n, exclude=("Other",)):
        u, c = np.unique(values, return_counts=True)
        order = np.argsort(-c)
        return [u[i] for i in order if u[i] not in exclude][:n]
