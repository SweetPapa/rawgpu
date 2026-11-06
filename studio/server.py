#!/usr/bin/env python3
"""rawgpu Studio: a local web app for training models on the GPU.

    python studio/server.py [--port 8765] [--no-open]       # then http://127.0.0.1:8765

Pick a recipe, set its parameters, start it, and watch it train: loss curves, evaluations, samples, the
card's temperature. Runs queue, since one program at a time can have the card; each runs in its own
process (studio/runner.py) and lives in a directory under <rawgpu data>/runs, so the server can stop and
start without losing anything. Use a Python with numpy (and, for the EmbeddingGemma 2 recipes,
tokenizers and huggingface_hub): this checkout's local/eg2-venv has them.

The server listens on 127.0.0.1 only, and accepts changes only as JSON from its own page.
"""
import argparse
import http.server
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run import ROOT, read_json, runs_dir, write_json  # noqa: E402
import recipes  # noqa: E402

UI = os.path.join(HERE, "ui")
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}
ID_RE = re.compile(r"^[a-z0-9-]{1,80}$")


def rawgpu_cli():
    for p in (os.path.join(ROOT, "build", "rawgpu"), shutil.which("rawgpu") or ""):
        if p and os.access(p, os.X_OK):
            return p
    return None


class Studio:
    def __init__(self, python):
        self.python = python
        self.lock = threading.Lock()
        self.proc = None        # the runner that has the card
        self.current = None     # its run id
        self.trying = False
        self._card = (0.0, None)
        self.recipes = recipes.catalog()
        self.recover()
        threading.Thread(target=self._monitor, daemon=True).start()

    # ---- runs
    def path(self, rid):
        if not ID_RE.match(rid or ""):
            raise KeyError(rid)
        p = os.path.join(runs_dir(), rid)
        if not os.path.isdir(p):
            raise KeyError(rid)
        return p

    def recover(self):
        """Runs left 'running' by a server that went away are not running any more"""
        for rid in os.listdir(runs_dir()):
            st = read_json(os.path.join(runs_dir(), rid, "status.json"))
            if st and st.get("state") == "running":
                pid = st.get("pid")
                alive = False
                if pid:
                    try:
                        os.kill(pid, 0)
                        alive = True
                    except OSError:
                        pass
                if not alive:
                    st.update(state="stopped", message="Interrupted; resume picks up from the last checkpoint")
                    write_json(os.path.join(runs_dir(), rid, "status.json"), st)

    def list_runs(self):
        out = []
        for rid in os.listdir(runs_dir()):
            d = os.path.join(runs_dir(), rid)
            cfg = read_json(os.path.join(d, "config.json"))
            if not cfg:
                continue
            st = read_json(os.path.join(d, "status.json"), {}) or {}
            out.append({"id": rid, "name": cfg.get("name") or rid, "recipe": cfg.get("recipe"),
                        "created": cfg.get("created", 0), "state": st.get("state", "queued"),
                        "step": st.get("step"), "total": st.get("total"), "message": st.get("message"),
                        "error": st.get("error"), "summary": st.get("summary", {}), "started": st.get("started"),
                        "finished": st.get("finished")})
        out.sort(key=lambda r: -r["created"])
        return out

    def create(self, recipe, params, name=None):
        meta = next((r for r in self.recipes if r["id"] == recipe), None)
        if not meta:
            raise ValueError(f"no recipe {recipe!r}")
        p = {}
        for spec in meta["params"]:
            v = params.get(spec["key"], spec["default"])
            t = spec["type"]
            try:
                if t == "int" or (t == "choice" and isinstance(spec["default"], int)):
                    v = int(v)
                elif t == "float":
                    v = float(v)
                elif t == "bool":
                    v = bool(v)
                else:
                    v = "" if v is None else str(v)
            except (TypeError, ValueError):
                raise ValueError(f"{spec['label']}: not a valid {t}")
            if t == "choice" and v not in spec["choices"]:
                raise ValueError(f"{spec['label']}: choose one of {spec['choices']}")
            if "min" in spec and v < spec["min"] or "max" in spec and v > spec["max"]:
                raise ValueError(f"{spec['label']}: between {spec.get('min')} and {spec.get('max')}")
            p[spec["key"]] = v
        now = time.time()
        rid = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + "-" + recipe.replace("_", "-")
        base, n = rid, 1
        while os.path.exists(os.path.join(runs_dir(), rid)):
            n += 1
            rid = f"{base}-{n}"
        d = os.path.join(runs_dir(), rid)
        os.makedirs(d)
        write_json(os.path.join(d, "config.json"), {"recipe": recipe, "name": (name or "").strip() or meta["name"],
                                                    "params": p, "created": now})
        write_json(os.path.join(d, "status.json"), {"state": "queued", "queued": now})
        self.schedule()
        return rid

    def set_state(self, rid, **kw):
        sp = os.path.join(self.path(rid), "status.json")
        st = read_json(sp, {}) or {}
        st.update(kw)
        write_json(sp, st)

    def schedule(self):
        with self.lock:
            if self.proc is not None or self.trying:
                return
            queued = [r for r in self.list_runs() if r["state"] == "queued"]
            if not queued:
                return
            run = min(queued, key=lambda r: r["created"])
            d = self.path(run["id"])
            st = read_json(os.path.join(d, "status.json"), {}) or {}
            args = [self.python, os.path.join(HERE, "runner.py"), d] + (["--resume"] if st.get("resume") else [])
            log = open(os.path.join(d, "log.txt"), "a")
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(args[1:])}\n")
            log.flush()
            env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
            self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env,
                                         start_new_session=True)
            self.current = run["id"]
            st.update(state="running", pid=self.proc.pid, message="Starting", resume=False)
            write_json(os.path.join(d, "status.json"), st)

    def _monitor(self):
        while True:
            time.sleep(1)
            with self.lock:
                p, rid = self.proc, self.current
            if p is not None and p.poll() is not None:
                st = read_json(os.path.join(runs_dir(), rid, "status.json"), {}) or {}
                if st.get("state") == "running":  # the runner died without saying why
                    st.update(state="failed", error=st.get("error") or f"the runner exited with code {p.returncode}"
                              " (see the log)", finished=time.time(), message="")
                    write_json(os.path.join(runs_dir(), rid, "status.json"), st)
                with self.lock:
                    self.proc, self.current = None, None
            self.schedule()

    def stop(self, rid):
        with self.lock:
            if rid == self.current and self.proc is not None:
                self.proc.send_signal(signal.SIGTERM)
                return "stopping"
        st = read_json(os.path.join(self.path(rid), "status.json"), {}) or {}
        if st.get("state") == "queued":
            self.set_state(rid, state="cancelled", message="")
            return "cancelled"
        return st.get("state")

    def resume(self, rid):
        st = read_json(os.path.join(self.path(rid), "status.json"), {}) or {}
        if st.get("state") in ("stopped", "failed", "cancelled", "done"):
            fresh = st.get("state") == "cancelled" or not os.path.exists(
                os.path.join(self.path(rid), "checkpoints", "latest.npz"))
            self.set_state(rid, state="queued", resume=not fresh, error=None, message="")
            self.schedule()
        return "queued"

    def delete(self, rid):
        d = self.path(rid)
        if rid == self.current:
            raise ValueError("stop the run first")
        if os.path.dirname(os.path.abspath(d)) != os.path.abspath(runs_dir()):
            raise ValueError("not a run directory")
        shutil.rmtree(d)

    def try_it(self, rid, req):
        d = self.path(rid)
        with self.lock:
            if self.proc is not None:
                raise ValueError("the card is busy with a training run; try again when it finishes")
            self.trying = True
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(req, f)
            r = subprocess.run([self.python, os.path.join(HERE, "runner.py"), d, "--try", f.name], capture_output=True,
                               text=True, timeout=300, cwd=ROOT, env=dict(os.environ, TOKENIZERS_PARALLELISM="false"))
            os.unlink(f.name)
            for line in r.stdout.splitlines():
                if line.startswith("__RESULT__"):
                    return json.loads(line[len("__RESULT__"):])
            tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
            raise ValueError("the playground failed: " + " / ".join(tail))
        finally:
            with self.lock:
                self.trying = False
            self.schedule()

    # ---- the card
    def card(self):
        with self.lock:
            rid = self.current
        if rid:
            st = read_json(os.path.join(runs_dir(), rid, "status.json"), {}) or {}
            return {"state": "training", "run": rid, **(st.get("card") or {})}
        t, cached = self._card
        if cached and time.time() - t < 15:
            return cached
        cli = rawgpu_cli()
        out = {"state": "unknown"}
        if not cli:
            out = {"state": "no-cli"}
        else:
            try:
                who = json.loads(subprocess.run([cli, "who", "--json"], capture_output=True, text=True, timeout=5).stdout)
                if not who.get("server_pid"):
                    out = {"state": "off"}
                elif who.get("client_count"):
                    out = {"state": "busy", "holder": (who.get("clients") or [{}])[0].get("name")}
                else:
                    st = json.loads(subprocess.run([cli, "status", "--json"], capture_output=True, text=True,
                                                   timeout=20).stdout)
                    out = {"state": "idle", **{k: st.get(k) for k in ("temperature_c", "engine_mhz", "memory_mhz",
                                                                      "fan_percent", "vram_mb", "posted")}}
            except Exception as e:
                out = {"state": "unknown", "error": str(e)}
        self._card = (time.time(), out)
        return out


def lines_from(path, start):
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return [], 0
    out = []
    for l in lines[start:]:
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return out, len(lines)


class Handler(http.server.BaseHTTPRequestHandler):
    studio = None
    port = 0
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _host_ok(self):
        host = (self.headers.get("Host") or "").lower()
        return host in (f"127.0.0.1:{self.port}", f"localhost:{self.port}")

    def _mutation_ok(self):
        if not self._host_ok():
            return False
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in (f"http://127.0.0.1:{self.port}", f"http://localhost:{self.port}")

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {"error": "forbidden"})
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        parts = [p for p in u.path.split("/") if p]
        S = self.studio
        try:
            if not parts or parts == ["index.html"]:
                return self._file("index.html")
            if parts[0] != "api":
                return self._file("/".join(parts))
            if parts[1:] == ["recipes"]:
                return self._send(200, S.recipes)
            if parts[1:] == ["runs"]:
                return self._send(200, {"runs": S.list_runs(), "current": S.current})
            if parts[1:] == ["card"]:
                return self._send(200, S.card())
            if len(parts) >= 3 and parts[1] == "runs":
                d = S.path(parts[2])
                if len(parts) == 3:
                    return self._send(200, {"id": parts[2], "config": read_json(os.path.join(d, "config.json")),
                                            "status": read_json(os.path.join(d, "status.json"), {})})
                what = parts[3]
                start = int(q.get("from", ["0"])[0])
                if what in ("metrics", "events"):
                    items, n = lines_from(os.path.join(d, what + ".jsonl"), start)
                    return self._send(200, {"items": items, "next": n})
                if what == "log":
                    try:
                        with open(os.path.join(d, "log.txt"), errors="replace") as f:
                            text = f.read()[-60000:]
                    except OSError:
                        text = ""
                    return self._send(200, {"text": text})
            return self._send(404, {"error": "not found"})
        except KeyError:
            return self._send(404, {"error": "no such run"})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        if not self._mutation_ok():
            return self._send(403, {"error": "forbidden"})
        parts = [p for p in urllib.parse.urlparse(self.path).path.split("/") if p]
        S = self.studio
        try:
            body = self._body()
            if parts == ["api", "runs"]:
                return self._send(200, {"id": S.create(body.get("recipe"), body.get("params") or {}, body.get("name"))})
            if len(parts) == 4 and parts[1] == "runs":
                rid, action = parts[2], parts[3]
                if action == "stop":
                    return self._send(200, {"state": S.stop(rid)})
                if action == "resume":
                    return self._send(200, {"state": S.resume(rid)})
                if action == "try":
                    return self._send(200, S.try_it(rid, body))
                if action == "rename":
                    d = S.path(rid)
                    cfg = read_json(os.path.join(d, "config.json"))
                    cfg["name"] = str(body.get("name", ""))[:120] or cfg["name"]
                    write_json(os.path.join(d, "config.json"), cfg)
                    return self._send(200, {"name": cfg["name"]})
            return self._send(404, {"error": "not found"})
        except KeyError:
            return self._send(404, {"error": "no such run"})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_DELETE(self):
        if not self._mutation_ok():
            return self._send(403, {"error": "forbidden"})
        parts = [p for p in urllib.parse.urlparse(self.path).path.split("/") if p]
        try:
            if len(parts) == 3 and parts[:2] == ["api", "runs"]:
                self.studio.delete(parts[2])
                return self._send(200, {"deleted": parts[2]})
            return self._send(404, {"error": "not found"})
        except KeyError:
            return self._send(404, {"error": "no such run"})
        except ValueError as e:
            return self._send(400, {"error": str(e)})

    def _file(self, rel):
        p = os.path.normpath(os.path.join(UI, rel))
        if not p.startswith(UI + os.sep) and p != os.path.join(UI, "index.html") or not os.path.isfile(p):
            return self._send(404, {"error": "not found"})
        with open(p, "rb") as f:
            body = f.read()
        return self._send(200, body, TYPES.get(os.path.splitext(p)[1], "application/octet-stream"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = ap.parse_args()
    Handler.studio = Studio(sys.executable)
    Handler.port = args.port
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"rawgpu Studio: {url}  (runs in {runs_dir()})", flush=True)
    if not args.no_open:
        subprocess.Popen(["open", url])
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
