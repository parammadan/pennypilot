#!/bin/bash
# One-shot V100 + deps confirmation (run via srun on a gpu node).
source ~/pennywise-v100-infra/env.sh
echo "NODE=$(hostname)  ENV=$CONDA_DEFAULT_ENV"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
python - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__, "| cuda_compiled", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "| cap", torch.cuda.get_device_capability(0))
    print("bf16_supported", torch.cuda.is_bf16_supported())
    free, total = torch.cuda.mem_get_info()
    print(f"mem_total_GB {total/1e9:.2f} | free_GB {free/1e9:.2f}")
print("transformers", transformers.__version__, "| peft", peft.__version__)
PY
