#!/bin/bash

echo "[1/7] lora_stack + adamw"
# uv run main.py adapters=lora_stack optimizer=adamw

echo "[2/7] lora_stack + muon"
uv run main.py adapters=lora_stack optimizer=muon

echo "[3/7] saft + adamw"
uv run main.py adapters=saft optimizer=adamw

echo "[4/7] saft + muon"
uv run main.py adapters=saft optimizer=muon


echo "[5/7] ella + muon"
uv run main.py adapters=ella optimizer=muon


echo "[6/7] olora + muon"
uv run main.py adapters=olora optimizer=muon


echo "[7/7] null_space + muon"
uv run main.py adapters=null_space optimizer=muon

echo "All 7 pipelines finished."
