#!/bin/bash

# Setup script for Spiritual Quest (SQ_Baha)
# This script sets up the backend and provides instructions for the mobile app.

set -e

# Check for Python
if ! command -v python3 &> /dev/null; then
  echo "[ERROR] Python3 is not installed. Please install Python 3.10 or higher."
  exit 1
fi

# Check for Node.js
if ! command -v node &> /dev/null; then
  echo "[ERROR] Node.js is not installed. Please install Node.js v16 or higher."
  exit 1
fi

# Check for pip
if ! command -v pip &> /dev/null; then
  echo "[ERROR] pip is not installed. Please install pip."
  exit 1
fi

# Backend setup
cd backend
if [ ! -d "../agent-env" ]; then
  echo "[INFO] Creating Python virtual environment..."
  python3 -m venv ../agent-env
fi
source ../agent-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd ..

echo "[INFO] Backend setup complete. To start the backend, run:"
echo "  cd backend && source ../agent-env/bin/activate && python3 main.py"

echo "[INFO] To run the mobile app:"
echo "  cd mobile && npm install && npm start"
echo "[INFO] Use Expo Go on your phone to scan the QR code."
echo "[INFO] Make sure to set the correct backend IP in App.js if needed."

echo "[INFO] Setup complete!" 