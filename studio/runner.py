#!/usr/bin/env python3
"""Run one Studio training run in this process (the server starts it; it can also run by hand):

    python studio/runner.py RUN_DIR [--resume]
    python studio/runner.py RUN_DIR --try REQUEST.json      # the recipe's playground, on a finished run

It holds the GPU for the whole run. SIGTERM (the Stop button) saves a checkpoint and exits.
"""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import Run, Stop, read_json  # noqa: E402
import recipes  # noqa: E402


def main():
    path = os.path.abspath(sys.argv[1])
    if "--try" in sys.argv:
        req = read_json(sys.argv[sys.argv.index("--try") + 1])
        cfg = read_json(os.path.join(path, "config.json"))
        mod = recipes.get(cfg["recipe"])
        run = Run.__new__(Run)  # no status writes, no signal handlers: just the paths and parameters
        run.dir, run.cfg, run.params = path, cfg, cfg["params"]
        out = mod.try_it(run, req)
        print("__RESULT__" + json.dumps(out), flush=True)
        return 0
    run = Run(path, resume="--resume" in sys.argv)
    mod = recipes.get(run.cfg["recipe"])
    run.set_status(state="running", pid=os.getpid(), started=time.time(), finished=None, error=None,
                   message="Opening the GPU" if not run.resume else "Resuming")
    code = 0
    try:
        from rawgpu import nn
        nn.ctx()
        run.set_status(message="")
        mod.train(run)
        run.set_status(state="done", finished=time.time(), message="")
    except Stop:
        if getattr(run, "saver", None):
            run.set_status(message="Saving a checkpoint")
            run.saver()
        run.set_status(state="stopped", finished=time.time(), message="Stopped; resume picks up from the last "
                                                                       "checkpoint")
    except Exception as e:
        traceback.print_exc()
        run.set_status(state="failed", finished=time.time(), error=f"{type(e).__name__}: {e}", message="")
        code = 1
    finally:
        try:
            from rawgpu import nn
            nn.close()
        except Exception:
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
