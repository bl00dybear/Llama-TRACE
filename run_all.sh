#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/10] lora_stack + adamw"
uv run main.py adapters=lora_stack optimizer=adamw

echo "[2/10] lora_stack + muon"
uv run main.py adapters=lora_stack optimizer=muon

echo "[3/10] saft + adamw"
uv run main.py adapters=saft optimizer=adamw

echo "[4/10] saft + muon"
uv run main.py adapters=saft optimizer=muon

echo "[5/10] ella + adamw"
uv run main.py adapters=ella optimizer=adamw

echo "[6/10] ella + muon"
uv run main.py adapters=ella optimizer=muon

echo "[7/10] olora + adamw"
uv run main.py adapters=olora optimizer=adamw

echo "[8/10] olora + muon"
uv run main.py adapters=olora optimizer=muon

echo "[9/10] null_space + adamw"
uv run main.py adapters=null_space optimizer=adamw

echo "[10/10] null_space + muon"
uv run main.py adapters=null_space optimizer=muon

echo "All 10 pipelines finished."
