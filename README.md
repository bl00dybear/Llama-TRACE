```bash
runpod pod create "Continuon-Llama-TRACE" \
  --gpu-count 1 \
  --gpu-type "NVIDIA RTX 3090" \
  --image "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
```