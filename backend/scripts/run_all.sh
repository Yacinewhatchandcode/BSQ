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
nohup uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 >/dev/null 2>&1 &
sleep 2
curl -sS http://127.0.0.1:8000/api/health || true

echo "[3/3] Running 160-variation smoke..."
python backend/variation_tester.py

echo "Done."

