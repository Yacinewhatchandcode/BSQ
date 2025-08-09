#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install -q -r requirements.txt || true
pip install -q playwright rich chromadb sentence-transformers || true
python -m playwright install --with-deps chromium || true

echo "[1/3] Running E2E..."
python backend/e2e_runner.py --headless

echo "[2/3] Starting server..."
uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 >/dev/null 2>&1 &
SERVER_PID=$!

# Wait for health up to 60s
for i in {1..60}; do
  if curl -sf http://127.0.0.1:8000/api/health >/dev/null; then
    echo "Server is healthy"
    break
  fi
  sleep 1
done

if ! curl -sf http://127.0.0.1:8000/api/health >/dev/null; then
  echo "Server failed to become healthy within timeout" >&2
  kill "$SERVER_PID" >/dev/null 2>&1 || true
  exit 1
fi

echo "[3/3] Running 160-variation smoke..."
python backend/variation_tester.py

# Cleanup: stop server so port remains closed
kill "$SERVER_PID" >/dev/null 2>&1 || true
sleep 1

echo "Done."

