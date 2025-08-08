#!/usr/bin/env bash
set -euo pipefail

# Defaults
APP_USER=${APP_USER:-"ubuntu"}
APP_DIR=${APP_DIR:-"/opt/spiritual-quest"}
PYTHON_BIN=${PYTHON_BIN:-"/usr/bin/python3"}
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-""}

# 1) Install base packages
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3 python3-venv python3-pip \
  curl unzip git

# 2) Create app directory
sudo mkdir -p "$APP_DIR"
sudo chown -R "$APP_USER":"$APP_USER" "$APP_DIR"
cd "$APP_DIR"

# 3) Unpack artifact if provided, else assume repo already synced
# Expect an archive named spiritual-quest-clean.zip near /tmp or APP_DIR
if [ -f /tmp/spiritual-quest-clean.zip ]; then
  rm -rf "$APP_DIR"/*
  unzip -q /tmp/spiritual-quest-clean.zip -d "$APP_DIR"
fi

# 4) Python venv
if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip >/dev/null 2>&1 || true
pip install -r requirements.txt >/dev/null 2>&1

# 5) Optional: install Playwright only if needed (skip browsers for server)
# pip install playwright >/dev/null 2>&1 || true

# 6) Environment file for service
sudo bash -c "cat > /etc/spiritual-quest.env <<EOF
OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
PORT=${PORT}
HOST=${HOST}
APP_DIR=${APP_DIR}
EOF"

# 7) Systemd unit
sudo bash -c "cat > /etc/systemd/system/spiritual-quest.service <<'SERVICE'
[Unit]
Description=Spiritual Quest FastAPI service
After=network.target

[Service]
Type=simple
EnvironmentFile=/etc/spiritual-quest.env
WorkingDirectory=%E{APP_DIR}
ExecStart=%E{APP_DIR}/.venv/bin/uvicorn main:app --app-dir %E{APP_DIR}/backend --host %E{HOST} --port %E{PORT}
User=%E{USER}
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
SERVICE"

# Replace %E{USER} & other env expansions not supported by default
sudo sed -i "s/%E{USER}/$APP_USER/g; s|%E{APP_DIR}|$APP_DIR|g; s/%E{HOST}/$HOST/g; s/%E{PORT}/$PORT/g" /etc/systemd/system/spiritual-quest.service

# 8) Reload & start
sudo systemctl daemon-reload
sudo systemctl enable spiritual-quest.service
sudo systemctl restart spiritual-quest.service

# 9) Status output
sleep 1
sudo systemctl --no-pager status spiritual-quest.service | head -n 50


