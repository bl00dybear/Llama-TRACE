```bash
runpod pod create \
  --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-ubuntu22.04" \
  --gpu-type "NVIDIA RTX 3090" \
  --gpu-count 1 \
  "Continuon-Llama-TRACE"
```