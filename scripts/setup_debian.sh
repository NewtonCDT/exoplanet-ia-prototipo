#!/usr/bin/env bash
set -euo pipefail
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
echo "✅ entorno listo"
