#!/usr/bin/env bash
# ==============================================================================
# Oracle Cloud (Ubuntu ARM/Ampere) one-shot setup for The Verifier.
#
# Installs Python + deps, clones the repo, pulls the model adapter from the HF
# Hub, and runs the Gradio app 24/7 as a systemd service bound to 127.0.0.1:7860.
# nginx (set up separately by setup_nginx.sh) puts it on port 80/443.
#
# Idempotent: safe to re-run. Run as the default 'ubuntu' user with sudo.
#
#   bash setup_oracle.sh
# ==============================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/oshikanoma/slm.git}"
APP_DIR="${APP_DIR:-/opt/verifier}"
SERVICE_USER="${SERVICE_USER:-ubuntu}"
HF_ADAPTER_REPO="${HF_ADAPTER_REPO:-tiffuhknee/qwen3-1.7b-newsroom-verifier}"
# Optional: open-web retrieval. Without it the app uses the free Wikipedia backend.
TAVILY_API_KEY="${TAVILY_API_KEY:-}"

echo "==> [1/6] System packages"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git nginx

echo "==> [2/6] Clone / update repo at ${APP_DIR}"
if [ -d "${APP_DIR}/.git" ]; then
  sudo git -C "${APP_DIR}" fetch --all --quiet
  sudo git -C "${APP_DIR}" reset --hard origin/main
else
  sudo git clone "${REPO_URL}" "${APP_DIR}"
fi
sudo chown -R "${SERVICE_USER}:${SERVICE_USER}" "${APP_DIR}"

echo "==> [3/6] Python venv + dependencies (CPU torch for ARM)"
cd "${APP_DIR}"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel
# CPU-only torch (the Oracle Ampere box has no GPU). The default index serves
# manylinux aarch64 CPU wheels for torch, so no special index URL is needed.
pip install torch --index-url https://download.pytorch.org/whl/cpu || pip install torch
pip install transformers peft accelerate gradio requests beautifulsoup4 pypdf python-docx huggingface_hub

echo "==> [4/6] Pre-download the model (so first visitor isn't the one who waits)"
HF_ADAPTER_REPO="${HF_ADAPTER_REPO}" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
base = "Qwen/Qwen3-1.7B"
adapter = os.environ["HF_ADAPTER_REPO"]
print(f"  downloading base {base} ...")
snapshot_download(base, allow_patterns=["*.json","*.safetensors","*.txt","tokenizer*","*.model"])
print(f"  downloading adapter {adapter} ...")
snapshot_download(adapter)
AutoTokenizer.from_pretrained(base)
print("  model cached.")
PY

echo "==> [5/6] Install systemd service"
sudo tee /etc/systemd/system/verifier.service >/dev/null <<UNIT
[Unit]
Description=The Verifier (Gradio) — Cited Newsroom Verifier SLM
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${APP_DIR}
Environment=HF_ADAPTER_REPO=${HF_ADAPTER_REPO}
Environment=GRADIO_SERVER_NAME=127.0.0.1
Environment=GRADIO_SERVER_PORT=7860
Environment=TAVILY_API_KEY=${TAVILY_API_KEY}
ExecStart=${APP_DIR}/.venv/bin/python ${APP_DIR}/app.py
Restart=always
RestartSec=5
# Model load is heavy; give it room and time.
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
UNIT

echo "==> [6/6] Start service"
sudo systemctl daemon-reload
sudo systemctl enable verifier
sudo systemctl restart verifier

echo
echo "Done. Service status:"
sudo systemctl --no-pager --full status verifier | head -12 || true
echo
echo "The app is now on 127.0.0.1:7860. Next: run setup_nginx.sh to expose it."
