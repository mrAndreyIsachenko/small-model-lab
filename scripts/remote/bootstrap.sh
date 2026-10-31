#!/usr/bin/env bash
# Bootstrap an ephemeral Ubuntu + NVIDIA GPU instance (RunPod / Lambda / Vast.ai)
set -euo pipefail

echo "[sml-bootstrap] Checking NVIDIA GPU availability..."
nvidia-smi

if ! command -v uv >/dev/null 2>&1; then
    echo "[sml-bootstrap] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "[sml-bootstrap] Syncing locked Python 3.12 environment..."
uv python install 3.12
uv sync --frozen --extra data --extra parity --extra telemetry

echo "[sml-bootstrap] Running fast pre-flight unit tests..."
uv run pytest

echo "[sml-bootstrap] Environment ready."
