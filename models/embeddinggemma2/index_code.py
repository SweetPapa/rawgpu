#!/usr/bin/env python3
"""Embed a folder of code (default ~/code) with EmbeddingGemma 2 on the RX 590, and search it.

  PY=local/eg2-venv/bin/python; IX=models/embeddinggemma2/index_code.py
  $PY $IX dry-run [--project NAME]...       # what would be indexed, skipped, and how long it would take
  caffeinate -i $PY $IX run [--project NAME]...
  $PY $IX search "how do we post the card" ["another query" ...] [-k 5] [--project NAME]

Safety:
  - The root is only read: files are opened read-only and nothing is written under it. The output
    folder (--out, default ~/rawgpu-index) must be outside the root.
  - No network: the model and tokenizer come from the local Hugging Face cache (HF_HUB_OFFLINE=1).
  - Other people's code is skipped (see third_party_reason()): a folder whose git remotes all belong
    to someone else, top-level or nested (submodules, clones inside a project); a folder whose package
    manifest (idf_component.yml, library.json, package.json, Cargo.toml) names someone else's
    repository; well-known open-source projects by name (llama.cpp, lvgl, openssl, ...); and vendored-
    code folders (upstream, third_party, external, managed_components, ...). Whose remotes count as
    yours comes from <out>/config.json: {"own_remotes": [regex, ...]}.
  - Skipped: .git and other VCS folders, node_modules, dist, build, target, virtualenvs, caches and
    other tool output; binaries, media, archives; minified and long-line data files; lockfiles;
    generated files; files over 1 MB; .env files; private keys and other credential files by name;
    and any file whose text looks like it holds a credential (key formats, tokens, passwords in URLs,
    `password = "..."`-style assignments).

Chunks: 200-500 tokens, cut where a cut reads naturally (a blank line, a top-level definition, a
heading). A cut that has to fall inside a block overlaps the next chunk by up to 40 tokens. Each is embedded as a document titled with its path:
"title: <project/path> | text: <chunk>".

Storage: one SQLite file, <out>/index.sqlite. Each row of `chunks` holds the path, project, first and
last line (1-based), the SHA-256 of the embedded text, the token count, the chunk text, and the
768-dimensional float32 vector. `files` records each indexed file's size and mtime. Work is committed
every --batch new chunks, so a run can stop at any point and pick up where it left off: files whose
size and mtime are unchanged are skipped without being read, and chunks whose hash is already in
the index reuse its vector.

A run checks itself: every --canary-minutes it embeds a fixed text and compares the vector with a
reference computed once by the reference implementation on the CPU, and it watches the card's
temperature. It stops on a mismatch, any GPU error, a non-finite vector, or --max-temp (85 C).
"""
import argparse
import collections
import hashlib
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"  # no network: the model must already be in the local cache
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MIN_TOK, TARGET_TOK, MAX_TOK, OVERLAP_TOK = 200, 350, 500, 40
MAX_FILE = 1 << 20
DIM = 768
CANARY = (
    "title: rawgpu/canary.md | text: # Canary\n\nThis fixed text is embedded every few minutes during a long "
    "indexing run, and its vector is compared with one the reference implementation computed on the CPU.\n\n"
    "```python\ndef saxpy(a, x, y):\n    return [a * xi + yi for xi, yi in zip(x, y)]\n```\n\n"
    "If the GPU, its memory or its clocks misbehave, this vector moves and the run stops.")

# ---------------------------------------------------------------- what to skip

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".bzr", "node_modules", "bower_components", "jspm_packages", "dist", "build", "target",
    "out", "venv", ".venv", "env", "virtualenv", "virtualenvs", ".virtualenvs", "site-packages", "__pycache__",
    ".tox", ".nox", ".eggs", ".next", ".nuxt", ".svelte-kit", ".turbo", ".gradle", ".idea", ".vscode", "coverage",
    ".nyc_output", "htmlcov", ".terraform", ".terragrunt-cache", "Pods", "Carthage", "DerivedData", ".expo",
    ".angular", ".dart_tool", ".pub-cache", "elm-stuff", "_build", "deps", ".stack-work", "zig-cache",
    "zig-out", "CMakeFiles", ".serverless", ".aws-sam", ".vercel", ".netlify", ".firebase", "storybook-static",
    ".ipynb_checkpoints", ".docusaurus", "vendor", "Library", "Temp", "Logs", "obj", "UserSettings", "Builds",
    ".pnpm-store", ".yarn", ".bundle", "tmp", "temp", "logs", ".history", ".DS_Store",
}
KEEP_DOT_DIRS = {".github", ".gitlab", ".circleci", ".devcontainer", ".storybook", ".husky", ".claude", ".cursor"}

BINARY_EXT = set("""
png jpg jpeg gif bmp ico icns tif tiff webp heic heif psd ai sketch fig xcf svg eps raw cr2 nef dng
mp3 wav ogg flac m4a aac aiff mid midi mp4 mov avi mkv webm m4v wmv flv 3gp
zip gz tgz bz2 xz 7z rar tar jar war ear dmg pkg iso img apk ipa aab xip zst lz4 cab msi deb rpm
exe dll so dylib a o obj lib bin class pyc pyo pyd wasm node elf out app framework
ttf otf woff woff2 eot fnt
pdf doc docx xls xlsx ppt pptx key numbers pages odt ods odp rtf epub
sqlite sqlite3 db db3 mdb accdb parquet feather arrow avro orc npy npz h5 hdf5 pkl pickle joblib pt pth ckpt
safetensors onnx tflite gguf ggml mlmodel caffemodel pb tfrecord bin rom trace
unitypackage asset prefab unity mat anim controller meta physicmaterial cubemap fbx blend obj3d glb gltf stl dae
3ds max ma mb usdz usd vox
swp swo bak tmp log lock map
""".split())
DATA_EXT = {"csv", "tsv", "jsonl", "ndjson", "dat", "sav", "dta"}
LOCKFILES = {
    "package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb", "bun.lock",
    "Cargo.lock", "poetry.lock", "Pipfile.lock", "uv.lock", "pdm.lock", "Gemfile.lock", "composer.lock", "go.sum",
    "Package.resolved", "flake.lock", "mix.lock", "pubspec.lock", "packages.lock.json", "project.assets.json",
    "Podfile.lock", "gradle.lockfile", "conda-lock.yml", "shrinkwrap.yaml",
}
KEY_EXT = {"pem", "key", "p12", "pfx", "jks", "keystore", "ppk", "p8", "der", "crt", "cer", "csr", "gpg", "pgp",
           "asc", "kdbx", "mobileprovision", "provisionprofile", "tfstate", "tfvars", "secret", "secrets"}
CREDENTIAL_NAMES = {
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "id_rsa.pub", "id_dsa.pub", "id_ecdsa.pub", "id_ed25519.pub",
    ".netrc", "_netrc", ".npmrc", ".pypirc", ".htpasswd", ".pgpass", ".git-credentials", "credentials",
    "credentials.json", "credentials.yml", "credentials.yaml", "token.json", "tokens.json", "auth.json",
    "kubeconfig", ".dockercfg", "config.json.enc", "google-services.json", "GoogleService-Info.plist",
    "known_hosts", "authorized_keys", ".boto", ".s3cfg", "secrets.json", "secrets.yml", "secrets.yaml",
    "secrets.toml", "secret.json", ".secrets", "master.key", "credentials.yml.enc", ".vault-token",
}
CREDENTIAL_NAME_RE = re.compile(
    r"(?i)(^|[._-])(client_secret|service[-_]?account|serviceaccount|firebase-adminsdk|private[-_]?key|"
    r"credentials?|secrets?)([._-]|$).*\.(json|ya?ml|toml|ini|cfg|conf|txt|xml|plist|properties)$")
GENERATED_NAME_RE = re.compile(
    r"(\.pb\.go|_pb2(_grpc)?\.pyi?|\.pb\.(h|cc|swift)|\.g\.dart|\.freezed\.dart|\.generated\.\w+|_generated\.\w+|"
    r"\.designer\.cs|\.Designer\.cs|\.min\.(js|css|mjs)|-min\.js|\.bundle\.js|\.chunk\.(js|css))$")
GENERATED_MARK_RE = re.compile(
    r"(?i)(@generated\b|do not edit|code generated by|auto-?generated|automatically generated|"
    r"this file (is|was) generated|generated by (the )?[a-z0-9_.-]+ (compiler|tool|plugin|generator))")

PLACEHOLDER_RE = re.compile(
    r"(?i)(example|sample|dummy|fake|placeholder|changeme|change_me|your[_-]|<|>|\$\{|\{\{|%\(|process\.env|"
    r"os\.environ|getenv|xxx|\*\*\*|redacted|^none$|^null$|^password$|^secret$|^todo|^test|test$|^[.]+$)")
CRED_STRONG = [re.compile(p) for p in (
    r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----",
    r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    r"(?i)aws_secret_access_key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}",
    r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
    r"\bgithub_pat_[A-Za-z0-9_]{50,}\b",
    r"\bglpat-[A-Za-z0-9_\-]{20,}\b",
    r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b",
    r"\bsk-ant-[A-Za-z0-9_\-]{20,}",
    r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{32,}\b",
    r"\bAIza[0-9A-Za-z_\-]{35}\b",
    r"\bya29\.[0-9A-Za-z_\-]{20,}",
    r"\b(?:sk|rk)_live_[0-9a-zA-Z]{20,}\b",
    r"\bhf_[A-Za-z0-9]{30,}\b",
    r"\bnpm_[A-Za-z0-9]{36}\b",
    r"\bpypi-[A-Za-z0-9_\-]{50,}\b",
    r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
)]
# a password inside a URL, and `secret = "value"`-style assignments: the value is checked for placeholders
CRED_URL = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s:/@'\"`]+:([^\s@/'\"`]{3,})@[^\s'\"`]+")
CRED_ASSIGN = re.compile(
    r"(?i)\b(?:password|passwd|pwd|pass|secret|secret[_-]?key|api[_-]?key|apikey|api[_-]?secret|access[_-]?key|"
    r"access[_-]?token|auth[_-]?token|refresh[_-]?token|client[_-]?secret|private[_-]?key|bearer|token)\b[\"']?"
    r"\s*[:=]\s*[\"']([^\"'\s]{8,})[\"']")


def credential_reason(text):
    for p in CRED_STRONG:
        if p.search(text):
            return "credential (key or token format)"
    for p, what in ((CRED_URL, "credential (password in a URL)"), (CRED_ASSIGN, "credential (secret assignment)")):
        for m in p.finditer(text):
            if not PLACEHOLDER_RE.search(m.group(1)):
                return what
    return None


def name_reason(name):
    """Why a file is skipped by its name alone, or None."""
    low = name.lower()
    ext = low.rsplit(".", 1)[-1] if "." in low[1:] else ""
    if low == ".env" or low.startswith(".env.") or low.endswith(".env") or low == ".envrc":
        return ".env file"
    if name in CREDENTIAL_NAMES or low.startswith("id_rsa") or low.startswith("id_ed25519") or ext in KEY_EXT \
            or CREDENTIAL_NAME_RE.search(name):
        return "private key or credential file"
    if name in LOCKFILES or ext == "lock":
        return "lockfile"
    if GENERATED_NAME_RE.search(name):
        return "minified" if ".min." in low or "-min." in low else "generated"
    if ext in BINARY_EXT:
        return "binary or media"
    if ext in DATA_EXT:
        return "data file"
    if name == ".DS_Store":
        return "binary or media"
    return None


# ---------------------------------------------------------------- other people's code

VENDOR_DIRS = {"upstream", "third_party", "third-party", "thirdparty", "3rdparty", "3rd_party", "3rd-party",
               "external", "externals", "extern", "vendored", "managed_components", "sourcepackages", "_deps"}
OSS_NAMES = {  # unambiguous names only: a folder called "linux" or "transformers" may well be your own
    "llama.cpp", "whisper.cpp", "ggml", "libyuv", "lvgl", "esp-idf", "openssl", "boringssl", "libressl", "mbedtls",
    "freertos-kernel", "cjson", "libjpeg-turbo", "libpng", "imgui", "googletest", "abseil-cpp", "pybind11",
    "nanobind", "tinygrad", "onnxruntime", "open-llm-vtuber", "hermes-agent", "chatterbox", "n8n-docs",
    "kompute", "llvm-project", "u-boot", "arduino-esp32", "esp-iot-solution", "esp-bsp", "esp-adf", "esp-who",
    "tflite-micro", "stable-diffusion-webui", "open-webui", "librechat", "openmanus", "open-interpreter",
    "browser-use", "firecrawl",
}
_cfg = {"own_remotes": []}


def load_config(out):
    """<out>/config.json: {"own_remotes": [regex, ...], "oss_names": [...], "vendor_dirs": [...]}"""
    path = os.path.join(out, "config.json")
    if os.path.exists(path):
        c = json.load(open(path))
        _cfg["own_remotes"] = [re.compile(r) for r in c.get("own_remotes", [])]
        OSS_NAMES.update(n.lower() for n in c.get("oss_names", []))
        VENDOR_DIRS.update(n.lower() for n in c.get("vendor_dirs", []))
    return path if os.path.exists(path) else None


def is_own(url):
    """A remote or repository URL that is the user's own. Relative URLs (submodules beside their parent)
    count as the parent's, so as own: a third-party parent is skipped on its own account."""
    if url.startswith(("./", "../")):
        return True
    return any(r.search(url) for r in _cfg["own_remotes"])


def git_remotes(path, depth=0):
    """The remote URLs of the git repository at path (a .git folder, or a .git file pointing to one), [] if
    it has none, or None if path is not a repository. A remote that is a local path is followed once."""
    g = os.path.join(path, ".git")
    if os.path.isfile(g):
        try:
            d = open(g).read().split("gitdir:", 1)[1].strip()
            g = d if os.path.isabs(d) else os.path.normpath(os.path.join(path, d))
        except (OSError, IndexError):
            return None
    try:
        txt = open(os.path.join(g, "config"), errors="replace").read()
    except OSError:
        return None
    urls = re.findall(r'\[remote "[^"]+"\][^\[]*?\burl\s*=\s*(\S+)', txt)
    out = []
    for u in urls:
        local = u[7:] if u.startswith("file://") else u
        if os.path.isabs(local) and depth == 0:
            out.extend(git_remotes(local[:-4] if local.endswith(".git") and not os.path.isdir(local) else local,
                                   depth + 1) or [])
        else:
            out.append(u)
    return out


MANIFEST_RE = {
    "idf_component.yml": re.compile(r"(?m)^(?:url|repository)\s*:\s*['\"]?(\S+?)['\"]?\s*$"),
    "library.json": re.compile(r'"url"\s*:\s*"([^"]+)"'),
    "package.json": re.compile(r'"repository"\s*:\s*(?:"([^"]+)"|\{[^}]*?"url"\s*:\s*"([^"]+)")'),
    "Cargo.toml": re.compile(r'(?m)^repository\s*=\s*"([^"]+)"'),
}


def manifest_repo(path):
    """The repository a package manifest in path names, or None."""
    for name, rx in MANIFEST_RE.items():
        f = os.path.join(path, name)
        if os.path.isfile(f):
            try:
                m = rx.search(open(f, errors="replace").read(65536))
            except OSError:
                continue
            if m:
                url = next(g for g in m.groups() if g)
                if "/" in url or ":" in url:
                    return url
    return None


PLACEHOLDER_OWNER = re.compile(r"(?i)[/:](your-?user-?name|your-?name|your-?org|username|user|owner|example|<[^>]*>)/")


def short_url(u):
    return re.sub(r"^(https?://|git@|git://|ssh://)", "", u).replace("github.com:", "github.com/")[:60]


def has_license(path):
    try:
        return any(re.match(r"(?i)(licen[cs]e|copying)(\.|$)", f) for f in os.listdir(path))
    except OSError:
        return False


def third_party_reason(name, path, top=False):
    """Why a folder is someone else's code, or None. Inside a project, a well-known name counts only
    with a license file beside it (a copy of the project, not a folder of yours named after it)."""
    low = name.lower()
    if low in VENDOR_DIRS:
        return f"vendored ({name}/)"
    base = re.sub(r"[-_.]?v?\d[\w.]*([-_].*)?$", "", low)
    if (low in OSS_NAMES or base in OSS_NAMES) and (top or has_license(path)):
        return f"known open-source project ({name})"
    if _cfg["own_remotes"]:
        urls = git_remotes(path)
        if urls and not any(is_own(u) for u in urls):
            return f"git remote {short_url(urls[0])}"
    url = manifest_repo(path)
    if url and _cfg["own_remotes"] and not is_own(url) and not PLACEHOLDER_OWNER.search(url):
        return f"manifest names {short_url(url)}"
    return None


def dir_reason(name, path, top=False):
    if name in KEEP_DOT_DIRS:
        return None
    if name in SKIP_DIRS or name.endswith(".egg-info") or name.startswith("cmake-build-"):
        return "skipped folder (" + name + ")"
    low = name.lower()
    if (name.startswith(".") or name.startswith("_")) and "cache" in low:
        return "skipped folder (cache)"
    if name.startswith("."):
        return "skipped folder (hidden)"
    if os.path.exists(os.path.join(path, "pyvenv.cfg")) or os.path.isdir(os.path.join(path, "conda-meta")):
        return "skipped folder (virtualenv)"
    why = third_party_reason(name, path, top)
    return "third-party: " + why if why else None


def read_text(path, size):
    """(text, reason): the file's text, or why it is skipped."""
    with open(path, "rb") as f:
        data = f.read(MAX_FILE + 1)
    if not data.strip():
        return None, "empty"
    if b"\0" in data[:8192]:
        return None, "binary or media"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
        if text.count("\ufffd") > len(text) / 200:
            return None, "binary or media"
    if path.endswith(".ipynb"):
        try:
            nb = json.loads(text)
            cells = nb.get("cells", [])
            text = "".join(f"# %% [{c.get('cell_type', 'code')}]\n" + "".join(c.get("source", [])).rstrip() + "\n\n"
                           for c in cells if c.get("cell_type") in ("code", "markdown"))
            if not text.strip():
                return None, "empty"
        except (ValueError, AttributeError):
            return None, "unreadable notebook"
    head = "\n".join(text[:1500].split("\n")[:10])  # generators mark the top of the file
    if GENERATED_MARK_RE.search(head):
        return None, "generated"
    lines = text.count("\n") + 1
    longest = max((len(l) for l in text.split("\n")), default=0)
    if longest > 5000 or (len(text) > 2000 and len(text) / lines > 300):
        return None, "minified or long-line data"
    why = credential_reason(text)
    if why:
        return None, why
    return text, None


def walk(root, projects=None, exclude=()):
    """Yield (relpath, project, abspath, stat, reason) for every file under root, in a stable order;
    reason is None for files to read, or why the file (or its folder) is skipped. Skipped folders are
    yielded once, as (relpath + "/", project, abspath, None, reason)."""
    def scan(path, rel, project):
        try:
            entries = sorted(os.scandir(path), key=lambda e: e.name)
        except OSError as e:
            yield rel + "/", project, path, None, "unreadable folder (" + e.__class__.__name__ + ")"
            return
        for e in entries:
            r = rel + "/" + e.name if rel else e.name
            proj = project or (e.name if e.is_dir(follow_symlinks=False) else "_root")
            if e.is_symlink():
                yield r, proj, e.path, None, "symlink"
            elif e.is_dir():
                if not rel and (projects and e.name not in projects or e.name in exclude):
                    continue
                why = dir_reason(e.name, e.path, top=not rel)
                if why:
                    yield r + "/", proj, e.path, None, why
                else:
                    yield from scan(e.path, r, proj)
            elif e.is_file():
                if not rel and (projects and "_root" not in projects or "_root" in exclude):
                    continue
                st = e.stat()
                why = name_reason(e.name) or ("over 1 MB" if st.st_size > MAX_FILE else None) or \
                    ("empty" if st.st_size == 0 else None)
                yield r, proj, e.path, st, why
    yield from scan(root, "", None)


# ---------------------------------------------------------------- chunking

DEF_RE = re.compile(r"(?:export\s+)?(?:default\s+)?(?:pub(?:\(\w+\))?\s+)?(?:async\s+)?(?:static\s+)?"
                    r"(?:def|class|function|func|fn|impl|struct|enum|trait|interface|type|module|namespace|"
                    r"package|object|protocol|extension|const|let|var|describe|it|test)\b")
def boundary_scores(lines):
    """score[j]: how natural a cut before line j is (j = len(lines): the end)."""
    n = len(lines)
    score = np.zeros(n + 1, np.float32)
    for j in range(1, n):
        cur, prev = lines[j], lines[j - 1]
        blank_prev = not prev.strip()
        top = bool(cur) and cur[0] not in " \t\r\n}])" and cur.strip() != ""
        s = 2.0 if blank_prev else 0.0
        if top:
            s += 1.0
            if DEF_RE.match(cur) or cur.startswith("@") or cur.startswith("#") or cur.startswith("/*") \
                    or cur.startswith("//") or cur.startswith("<!--"):
                s += 2.0
        elif prev.rstrip().endswith(("}", "end", "];", ");")) and blank_prev:
            s += 0.5
        score[j] = s
    score[n] = 10.0
    return score


def chunk_lines(lines, ntok):
    """Cut lines (with their per-line token counts) into chunks of MIN_TOK..MAX_TOK tokens where a cut
    reads naturally. A cut inside a block (not at a top-level line after a blank one, or a heading)
    overlaps the next chunk by up to OVERLAP_TOK tokens. Returns [(first line, end line exclusive,
    piece or None)]: piece is (char start, char end) when one long line is split."""
    n = len(lines)
    cum = np.concatenate([[0], np.cumsum(ntok)]).astype(np.int64)
    score = boundary_scores(lines)
    out, s = [], 0
    while s < n:
        if cum[n] - cum[s] <= MAX_TOK:
            out.append((s, n, None))
            break
        hi = int(np.searchsorted(cum, cum[s] + MAX_TOK, side="right")) - 1  # longest end that fits
        if hi <= s:  # line s alone is longer than MAX_TOK: split it by characters
            L = len(lines[s])
            parts = int(np.ceil(ntok[s] / TARGET_TOK))
            step = max(1, L // parts)
            for a in range(0, L, step):
                out.append((s, s + 1, (a, min(L, a + step))))
            s += 1
            continue
        lo = int(np.searchsorted(cum, cum[s] + MIN_TOK, side="left"))
        if lo > hi:
            e = hi
        else:
            cand = np.arange(lo, hi + 1)
            toks = cum[cand] - cum[s]
            value = score[cand] - 3.0 * np.abs(toks - TARGET_TOK) / TARGET_TOK
            e = int(cand[np.argmax(value)])
        out.append((s, e, None))
        b = e
        if score[e] < 3:  # a cut inside a block: overlap; at a clean boundary the next chunk starts on it
            while b - 1 > s and cum[e] - cum[b - 1] <= OVERLAP_TOK:
                b -= 1
        s = b if b > s else e
    return out


def chunk_file(tok, text):
    """[(start line 1-based, end line 1-based, chunk text)] for one file's text."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    starts = np.cumsum([0] + [len(l) for l in lines[:-1]])
    enc = tok.encode(text, add_special_tokens=False)
    tstarts = np.fromiter((o[0] for o in enc.offsets), np.int64, len(enc.offsets))
    ntok = np.bincount(np.searchsorted(starts, tstarts, side="right") - 1, minlength=len(lines))[:len(lines)]
    chunks = []
    for s, e, piece in chunk_lines(lines, ntok):
        body = lines[s][piece[0]:piece[1]] if piece else "".join(lines[s:e])
        if body.strip():
            chunks.append((s + 1, e, body.rstrip() + "\n"))
    return chunks


# ---------------------------------------------------------------- storage

SCHEMA = """
CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, project TEXT NOT NULL, size INTEGER, mtime_ns INTEGER,
                                 chunks INTEGER, indexed_at REAL);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, path TEXT NOT NULL, project TEXT NOT NULL,
                                  start_line INTEGER NOT NULL, end_line INTEGER NOT NULL, hash TEXT NOT NULL,
                                  tokens INTEGER NOT NULL, text TEXT NOT NULL, vec BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE INDEX IF NOT EXISTS chunks_hash ON chunks(hash);
CREATE INDEX IF NOT EXISTS chunks_project ON chunks(project);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


def open_db(out):
    db = sqlite3.connect(os.path.join(out, "index.sqlite"))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(SCHEMA)
    db.execute("INSERT OR IGNORE INTO meta VALUES ('format', ?)", (json.dumps(dict(
        model="google/embeddinggemma-2", dim=DIM, dtype="float32", min_tok=MIN_TOK, max_tok=MAX_TOK,
        overlap_tok=OVERLAP_TOK, title="title: <path> | text: <chunk>")),))
    db.commit()
    return db


def snapshot():
    from eg2 import _snapshot
    return _snapshot("google/embeddinggemma-2")


def load_tokenizer():
    from tokenizers import Tokenizer
    path = snapshot()
    cfg = json.load(open(os.path.join(path, "config.json")))["text_config"]
    return Tokenizer.from_file(os.path.join(path, "tokenizer.json")), cfg["bos_token_id"], cfg["eos_token_id"]


def final_ids(tok, bos, eos, texts, max_seq=8192):
    """Token ids as EmbeddingGemma2.tokenize() makes them (no prompt: the title is in the text)."""
    return [[bos] + e.ids[:max_seq - 2] + [eos] for e in tok.encode_batch(texts, add_special_tokens=False)]


def embed_text(rel, chunk):
    return f"title: {rel} | text: {chunk}"


# ---------------------------------------------------------------- the scan both dry-run and run share

class Scan:
    """Walks the root, reads and filters files, and chunks them, counting what it skips."""

    def __init__(self, root, projects, tok, known=None, exclude=()):
        self.root, self.projects, self.tok, self.exclude = root, projects, tok, exclude or ()
        self.known = known or {}  # path -> (size, mtime_ns) already indexed
        self.skipped = collections.Counter()
        self.skipped_bytes = collections.Counter()
        self.skipped_dirs = collections.Counter()
        self.skipped_examples = collections.defaultdict(list)
        self.errors = []
        self.third_party = []
        self.seen = set()
        self.files = self.unchanged = 0

    def skip(self, rel, reason, size=0, folder=False):
        (self.skipped_dirs if folder else self.skipped)[reason] += 1
        self.skipped_bytes[reason] += size
        if len(self.skipped_examples[reason]) < 3:
            self.skipped_examples[reason].append(rel)

    def __iter__(self):
        """Yield (rel, project, stat, chunks) for files to (re)index; unchanged files are only noted."""
        for rel, project, path, st, why in walk(self.root, self.projects, self.exclude):
            if st is None:
                self.skip(rel, why, folder=rel.endswith("/"))
                if why.startswith("third-party"):
                    self.third_party.append((rel, why))
                continue
            if why:
                self.skip(rel, why, st.st_size)
                continue
            self.seen.add(rel)
            if self.known.get(rel) == (st.st_size, st.st_mtime_ns):
                self.unchanged += 1
                continue
            try:
                text, why = read_text(path, st.st_size)
            except OSError as e:
                self.errors.append(f"{rel}: {e.__class__.__name__}: {e}")
                self.seen.discard(rel)
                continue
            if why:
                self.skip(rel, why, st.st_size)
                self.seen.discard(rel)
                continue
            self.files += 1
            yield rel, project, st, chunk_file(self.tok, text)


def report_skips(scan, out=sys.stdout):
    tp = [(rel, why) for rel, why in scan.third_party]
    if tp:
        print(f"other people's code, not indexed ({len(tp)} folders):", file=out)
        for rel, why in tp:
            print(f"  {rel:70s} {why[len('third-party: '):]}", file=out)
    print("skipped folders (not entered):", file=out)
    for why, n in scan.skipped_dirs.most_common():
        if not why.startswith("third-party"):
            print(f"  {n:8d}  {why}", file=out)
    print("skipped files:", file=out)
    for why, n in scan.skipped.most_common():
        ex = ", ".join(scan.skipped_examples[why][:2])
        print(f"  {n:8d}  {why:38s} {scan.skipped_bytes[why] / 2**20:9.1f} MB   e.g. {ex}", file=out)
    if scan.errors:
        print(f"read errors: {len(scan.errors)}", file=out)
        for e in scan.errors[:10]:
            print("  " + e, file=out)


def cmd_dry_run(args):
    if not load_config(args.out):
        print(f"note: no {os.path.join(args.out, 'config.json')}, so git remotes and manifests are not used to "
              "tell your code from other people's")
    tok, bos, eos = load_tokenizer()
    scan = Scan(args.root, args.project, tok, exclude=args.exclude)
    t0 = time.time()
    per_project = collections.defaultdict(lambda: [0, 0, 0])  # files, chunks, tokens
    nchunks = ntokens = text_bytes = 0
    sizes = []
    pending = []

    def count(pending):
        nonlocal ntokens
        ids = final_ids(tok, bos, eos, [t for _, t in pending])
        for (proj, _), s in zip(pending, ids):
            per_project[proj][2] += len(s)
            ntokens += len(s)
            sizes.append(len(s))

    for rel, project, st, chunks in scan:
        per_project[project][0] += 1
        per_project[project][1] += len(chunks)
        nchunks += len(chunks)
        for s, e, c in chunks:
            text_bytes += len(c.encode())
            pending.append((project, embed_text(rel, c)))
        if len(pending) >= 4096:
            count(pending)
            pending = []
        if scan.files % 5000 == 0 and scan.files:
            print(f"  ... {scan.files} files, {nchunks} chunks, {time.time() - t0:.0f} s", flush=True)
    if pending:
        count(pending)
    dt = time.time() - t0

    print(f"\nroot: {args.root}{' (projects: ' + ', '.join(args.project) + ')' if args.project else ''}"
          f"{' (excluding: ' + ', '.join(args.exclude) + ')' if args.exclude else ''}")
    print(f"files to index:  {scan.files}")
    print(f"chunks:          {nchunks}  (tokens per chunk: median {int(np.median(sizes)) if sizes else 0}, "
          f"90th percentile {int(np.percentile(sizes, 90)) if sizes else 0}, max {max(sizes) if sizes else 0}; "
          f"with title and the BOS/EOS tokens)")
    print(f"tokens:          {ntokens}")
    gpu_h = ntokens / args.rate / 3600
    vec_mb = nchunks * DIM * 4 / 2**20
    db_mb = (nchunks * (DIM * 4 + 120) + text_bytes) * 1.15 / 2**20
    print(f"estimated time:  {gpu_h:.1f} h of GPU time at {args.rate:.0f} tokens/s, plus host time "
          f"(this scan took {dt / 60:.1f} min)")
    print(f"estimated size:  {db_mb / 1024:.2f} GB ({vec_mb / 1024:.2f} GB of vectors, "
          f"{text_bytes / 2**30:.2f} GB of chunk text, plus indexes)")
    print("\nlargest projects by tokens:")
    for p, (f, c, t) in sorted(per_project.items(), key=lambda kv: -kv[1][2])[:25]:
        print(f"  {t:12d} tokens  {c:8d} chunks  {f:7d} files  {p}")
    print(f"  ({len(per_project)} projects in all)\n")
    report_skips(scan)


# ---------------------------------------------------------------- run

class Stop(Exception):
    pass


class Runner:
    def __init__(self, args):
        self.args = args
        self.out = args.out
        os.makedirs(os.path.join(self.out, "logs"), exist_ok=True)
        self.db = open_db(self.out)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_path = os.path.join(self.out, "logs", f"run-{stamp}.jsonl")
        self.log_f = open(self.log_path, "a")
        self.t0 = time.time()
        self.gpu_s = self.enc_host_s = 0.0
        self.embedded = self.reused = self.tokens = self.batches = 0
        self.ranges = collections.defaultdict(lambda: [None, None])
        self.last_print = 0.0
        self.last_canary = 0.0

    def log(self, kind, **kw):
        rec = dict(t=round(time.time(), 3), elapsed_s=round(time.time() - self.t0, 1), kind=kind, **kw)
        self.log_f.write(json.dumps(rec) + "\n")
        self.log_f.flush()
        return rec

    def say(self, msg):
        print(time.strftime("%H:%M:%S ") + msg, flush=True)

    # -- the card's state
    def card(self):
        i = self.model.gpu.info
        st = {k: i[k] for k in ("engine_mhz", "memory_mhz", "temperature_c", "fan_percent")}
        for k, v in st.items():
            if v is not None and v >= 0:
                r = self.ranges[k]
                r[0] = v if r[0] is None else min(r[0], v)
                r[1] = v if r[1] is None else max(r[1], v)
        if st["temperature_c"] >= self.args.max_temp:
            self.log("card", **st)
            raise Stop(f"temperature {st['temperature_c']} C reached the limit of {self.args.max_temp} C")
        return st

    def canary(self, reason):
        ids = final_ids(self.tok, self.bos, self.eos, [CANARY])
        v = self.model.encode_tokens(ids)[0]
        cos = float(v @ self.canary_ref)
        diff = float(np.abs(v - self.canary_ref).max())
        ok = bool(np.isfinite(v).all()) and cos >= 0.99999 and diff <= 1e-5
        self.log("canary", reason=reason, cosine=cos, max_abs_diff=diff, ok=ok, **self.card())
        self.last_canary = time.time()
        if not ok:
            raise Stop(f"canary mismatch: cosine {cos:.7f}, max |difference| {diff:.2e}")
        return cos, diff

    def reference_canary(self):
        path = os.path.join(self.out, "canary.json")
        if os.path.exists(path):
            ref = json.load(open(path))
            if ref["text"] == CANARY:
                return np.array(ref["vector"], np.float32)
        self.say("computing the canary's reference vector on the CPU (sentence-transformers) ...")
        code = ("import json, sys, warnings; warnings.filterwarnings('ignore'); import torch; "
                "from sentence_transformers import SentenceTransformer; "
                "m = SentenceTransformer('google/embeddinggemma-2', device='cpu', "
                "model_kwargs={'torch_dtype': torch.float32}); "
                "t = sys.stdin.read(); v = m.encode([t], normalize_embeddings=True)[0]; "
                "print(json.dumps([float(x) for x in v]))")
        r = subprocess.run([sys.executable, "-c", code], input=CANARY, capture_output=True, text=True,
                           env=dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false"))
        if r.returncode:
            raise Stop("could not compute the canary reference: " + r.stderr[-500:])
        vec = json.loads(r.stdout.strip().splitlines()[-1])
        json.dump(dict(text=CANARY, vector=vec, by="sentence-transformers, float32, CPU"), open(path, "w"))
        return np.array(vec, np.float32)

    # -- the work
    def flush(self, pending):
        """Embed the pending files' new chunks and commit them, with the files' rows, in one transaction."""
        if not pending:
            return
        todo, seen = [], {}
        for f in pending:
            for c in f["chunks"]:
                if c["vec"] is None and c["hash"] not in seen:
                    seen[c["hash"]] = len(todo)
                    todo.append(c)
        if todo:
            ids = final_ids(self.tok, self.bos, self.eos, [c["etext"] for c in todo])
            vecs, st = self.model.encode_tokens(ids, return_stats=True)
            if not np.isfinite(vecs).all() or np.abs(np.linalg.norm(vecs, axis=1) - 1).max() > 1e-3:
                raise Stop("the GPU returned non-finite or non-unit vectors")
            for c, s in zip(todo, ids):
                c["tokens"] = len(s)
            self.gpu_s += st["gpu_s"]
            self.enc_host_s += st["host_s"]
            self.tokens += st["tokens"]
            for f in pending:
                for c in f["chunks"]:
                    if c["vec"] is None:
                        j = seen[c["hash"]]
                        c["vec"] = vecs[j].tobytes()
                        c["tokens"] = todo[j]["tokens"]
        self.embedded += len(todo)
        with self.db:
            for f in pending:
                self.db.execute("DELETE FROM chunks WHERE path = ?", (f["rel"],))
                self.db.executemany(
                    "INSERT INTO chunks(path, project, start_line, end_line, hash, tokens, text, vec) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [(f["rel"], f["project"], c["start"], c["end"], c["hash"], c["tokens"], c["text"], c["vec"])
                     for c in f["chunks"]])
                self.db.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?)",
                                (f["rel"], f["project"], f["size"], f["mtime_ns"], len(f["chunks"]), time.time()))
        self.batches += 1

    def progress(self, scan, force=False):
        now = time.time()
        if not force and now - self.last_print < self.args.log_seconds:
            return
        self.last_print = now
        wall = now - self.t0
        st = self.card()
        host = wall - self.gpu_s
        rec = self.log("progress", files_read=scan.files, files_unchanged=scan.unchanged,
                       chunks_embedded=self.embedded, chunks_reused=self.reused, tokens=self.tokens,
                       chunks_per_s=round(self.embedded / wall, 2), tokens_per_s=round(self.tokens / wall),
                       gpu_s=round(self.gpu_s, 1), host_s=round(host, 1),
                       host_share=round(host / wall, 3) if wall else 0, errors=len(scan.errors), **st)
        fan = f" fan {st['fan_percent']}%" if st["fan_percent"] >= 0 else ""
        self.say(f"{scan.files} files, {self.embedded} chunks embedded ({rec['chunks_per_s']:.1f}/s, "
                 f"{rec['tokens_per_s']} tok/s), {self.reused} reused | {st['engine_mhz']}/{st['memory_mhz']} MHz "
                 f"{st['temperature_c']} C{fan} | host {100 * rec['host_share']:.0f}% of the time | "
                 f"errors {len(scan.errors)}")

    def run(self):
        args = self.args
        if not load_config(self.out):
            self.say(f"note: no {os.path.join(self.out, 'config.json')}, so git remotes and manifests are not used "
                     "to tell your code from other people's")
        self.tok, self.bos, self.eos = load_tokenizer()
        from eg2 import EmbeddingGemma2
        self.canary_ref = self.reference_canary()
        self.say("loading the model onto the RX 590 ...")
        self.model = EmbeddingGemma2(verbose=False)
        self.canary("start")
        known = {p: (s, m) for p, s, m in self.db.execute("SELECT path, size, mtime_ns FROM files")}
        scan = Scan(args.root, args.project, self.tok, known, args.exclude)
        self.log("start", root=args.root, projects=args.project, known_files=len(known), pid=os.getpid())
        self.say(f"indexing {args.root}{' (' + ', '.join(args.project) + ')' if args.project else ''} into "
                 f"{self.out}; {len(known)} files already indexed")
        pending, npending, stop = [], 0, None
        try:
            for rel, project, st, chunks in scan:
                old = {h: (v, n) for h, v, n in
                       self.db.execute("SELECT hash, vec, tokens FROM chunks WHERE path = ?", (rel,))}
                f = dict(rel=rel, project=project, size=st.st_size, mtime_ns=st.st_mtime_ns, chunks=[])
                for s, e, text in chunks:
                    et = embed_text(rel, text)
                    h = hashlib.sha256(et.encode()).hexdigest()
                    vec, ntok = old.get(h) or self.db.execute(
                        "SELECT vec, tokens FROM chunks WHERE hash = ? LIMIT 1", (h,)).fetchone() or (None, 0)
                    if vec is not None:
                        self.reused += 1
                    else:
                        npending += 1
                    f["chunks"].append(dict(start=s, end=e, text=text, etext=et, hash=h, vec=vec, tokens=ntok))
                pending.append(f)
                if npending >= args.batch or len(pending) >= 4 * args.batch:
                    self.flush(pending)
                    pending, npending = [], 0
                    self.progress(scan)
                    if time.time() - self.last_canary >= args.canary_minutes * 60:
                        self.canary("periodic")
            self.flush(pending)
            pending = []
            # files that are gone, or now skipped, within the projects this run walked
            def walked(p):
                proj = p.split("/")[0] if "/" in p else "_root"
                return (not args.project or proj in args.project) and proj not in (args.exclude or ())
            stale = [p for (p,) in self.db.execute("SELECT path FROM files") if p not in scan.seen and walked(p)]
            with self.db:
                for p in stale:
                    self.db.execute("DELETE FROM chunks WHERE path = ?", (p,))
                    self.db.execute("DELETE FROM files WHERE path = ?", (p,))
            self.canary("end")
        except Stop as e:
            stop = str(e)
        except KeyboardInterrupt:
            stop = "interrupted"
        except Exception as e:  # a GPU fault or anything else: stop, keep what is committed
            import traceback
            stop = f"{e.__class__.__name__}: {e}"
            self.log("error", error=stop, traceback=traceback.format_exc())
        try:
            self.progress(scan, force=True)
        except Exception:
            pass
        wall = time.time() - self.t0
        n_chunks, n_files = self.db.execute("SELECT COUNT(*), COUNT(DISTINCT path) FROM chunks").fetchone()
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        size = sum(os.path.getsize(os.path.join(self.out, f)) for f in os.listdir(self.out)
                   if f.startswith("index.sqlite"))
        summary = self.log("end", ok=stop is None, stop_reason=stop, wall_s=round(wall, 1), gpu_s=round(self.gpu_s, 1),
                           host_s=round(wall - self.gpu_s, 1), files_read=scan.files, files_unchanged=scan.unchanged,
                           chunks_embedded=self.embedded, chunks_reused=self.reused, tokens=self.tokens,
                           index_chunks=n_chunks, index_files=n_files, index_bytes=size,
                           ranges={k: v for k, v in self.ranges.items()}, errors=scan.errors[:200],
                           skipped=dict(scan.skipped), skipped_dirs=dict(scan.skipped_dirs))
        self.say(("STOPPED: " + stop) if stop else "done")
        self.say(f"{wall / 3600:.2f} h; {self.embedded} chunks embedded, {self.reused} reused; index now "
                 f"{n_chunks} chunks from {n_files} files, {size / 2**30:.2f} GB; GPU {self.gpu_s / max(wall, 1e-9):.0%} "
                 f"of the time; ranges {json.dumps(summary['ranges'])}; {len(scan.errors)} read errors")
        self.say(f"log: {self.log_path}")
        report_skips(scan)
        return 0 if stop is None else 2


def cmd_run(args):
    if not os.environ.get("RAWGPU_NO_CAFFEINATE"):  # keep the Mac awake: sleep resets the card
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    def on_term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)  # stop like Ctrl-C: keep what is committed, close the card
    return Runner(args).run()


# ---------------------------------------------------------------- search

def vectors(out, db):
    """All vectors as one float32 matrix (cached next to the index) and their row ids."""
    n, top = db.execute("SELECT COUNT(*), MAX(id) FROM chunks").fetchone()
    stamp = f"{n}-{top}-{os.path.getmtime(os.path.join(out, 'index.sqlite'))}"
    vpath, ipath, spath = (os.path.join(out, f) for f in ("search-vectors.npy", "search-ids.npy", "search-stamp"))
    if os.path.exists(spath) and open(spath).read() == stamp:
        return np.load(vpath, mmap_mode="r"), np.load(ipath)
    ids = np.empty(n, np.int64)
    V = np.lib.format.open_memmap(vpath, mode="w+", dtype=np.float32, shape=(n, DIM))
    for i, (rid, vec) in enumerate(db.execute("SELECT id, vec FROM chunks ORDER BY id")):
        ids[i] = rid
        V[i] = np.frombuffer(vec, np.float32)
    V.flush()
    np.save(ipath, ids)
    open(spath, "w").write(stamp)
    return np.load(vpath, mmap_mode="r"), ids


def cmd_search(args):
    db = sqlite3.connect(os.path.join(args.out, "index.sqlite"))
    V, ids = vectors(args.out, db)
    if args.project:
        keep = {r for (r,) in db.execute(f"SELECT id FROM chunks WHERE project IN ({','.join('?' * len(args.project))})",
                                         args.project)}
        mask = np.fromiter((i in keep for i in ids), bool, len(ids))
    else:
        mask = None
    from eg2 import EmbeddingGemma2
    model = EmbeddingGemma2(verbose=False)
    Q = model.encode(args.queries, prompt=args.task)
    for q, qv in zip(args.queries, Q):
        with np.errstate(all="ignore"):  # Accelerate's sgemv raises spurious FP flags
            s = np.asarray(V @ qv)
        if mask is not None:
            s = np.where(mask, s, -2.0)
        top = np.argsort(-s)[:args.k]
        print(f"\n> {q}")
        for j in top:
            path, a, b, text = db.execute("SELECT path, start_line, end_line, text FROM chunks WHERE id = ?",
                                          (int(ids[j]),)).fetchone()
            first = " ".join(text.split())[:110]
            print(f"  {s[j]:.3f}  {path}:{a}-{b}\n         {first}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.expanduser("~/code"))
    ap.add_argument("--out", default=os.path.expanduser("~/rawgpu-index"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dry-run", "run", "search"):
        p = sub.add_parser(name)
        p.add_argument("--project", action="append", help="only these top-level folders (repeatable)")
        p.add_argument("--exclude", action="append", default=[], help="skip these top-level folders (repeatable)")
        if name == "dry-run":
            p.add_argument("--rate", type=float, default=10000, help="tokens per second, for the estimate")
        if name == "run":
            p.add_argument("--batch", type=int, default=1024, help="new chunks per commit")
            p.add_argument("--canary-minutes", type=float, default=10)
            p.add_argument("--max-temp", type=int, default=85)
            p.add_argument("--log-seconds", type=float, default=60)
        if name == "search":
            p.add_argument("queries", nargs="+")
            p.add_argument("-k", type=int, default=5)
            p.add_argument("--task", default="code", help="query prompt: code, query, qa, ... (eg2.PROMPTS)")
    args = ap.parse_args()
    args.root = os.path.realpath(args.root)
    args.out = os.path.realpath(args.out)
    if args.out == args.root or args.out.startswith(args.root + os.sep):
        sys.exit(f"--out ({args.out}) must be outside the folder being indexed ({args.root})")
    if args.cmd != "dry-run":
        os.makedirs(args.out, exist_ok=True)
    return {"dry-run": cmd_dry_run, "run": cmd_run, "search": cmd_search}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
