#!/bin/bash
# Start rawgpu Studio, the training web app, and open it in the browser.
#
#   tools/studio.sh [--port 8765] [--no-open]
#
# It needs a Python with numpy; the EmbeddingGemma 2 recipes also need tokenizers and huggingface_hub.
# This checkout's local/eg2-venv has all three (models/embeddinggemma2/README.md sets it up); otherwise
# $PYTHON, or python3.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/local/eg2-venv/bin/python" ]; then PY="$ROOT/local/eg2-venv/bin/python"; else PY=python3; fi
fi
if ! "$PY" -c "import numpy" 2>/dev/null; then
  echo "rawgpu Studio needs numpy: $PY -m pip install numpy (and tokenizers huggingface_hub for the EmbeddingGemma 2 recipes)" >&2
  exit 1
fi
[ -e "$ROOT/build/librawgpu.dylib" ] || { echo "build rawgpu first: make -C $ROOT" >&2; exit 1; }
exec "$PY" "$ROOT/studio/server.py" "$@"
