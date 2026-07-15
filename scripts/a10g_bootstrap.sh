#!/bin/bash
# Bootstrap a g5.xlarge (A10G, DLAMI Ubuntu 22.04) for vLLM rollout.
# The instance IAM role grants S3 read on the bucket -> no creds needed here.
set -euo pipefail
BUCKET=${BUCKET:-pennywise-794528241070-us-east-2}
RG=${RG:-us-east-2}
CKPT_PREFIX=${CKPT_PREFIX:-checkpoints/rloo50/policy}

cd ~
mkdir -p pennywise && cd pennywise

echo "[bootstrap] pulling code + checkpoint from S3 ..."
aws s3 cp s3://$BUCKET/code/pennywise-src.tar.gz . --region $RG
tar xzf pennywise-src.tar.gz
aws s3 sync s3://$BUCKET/$CKPT_PREFIX/ ./ckpt/ --region $RG

echo "[bootstrap] installing vllm + client deps (DLAMI already has CUDA driver) ..."
pip install -q --upgrade pip
pip install -q vllm httpx
pip install -q -e .

echo "[bootstrap] GPU check:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
python -c "import vllm; print('vllm', vllm.__version__)"
echo "BOOTSTRAP_DONE"

cat <<'NEXT'
--- next (run manually / via ssh) ---
# start the vLLM server (background):
nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/pennywise/ckpt --served-model-name pennywise-rloo \
  --dtype bfloat16 --max-model-len 2048 --gpu-memory-utilization 0.9 \
  > ~/pennywise/vllm.log 2>&1 &
# wait for "Application startup complete" in vllm.log, then:
python scripts/rollout_vllm.py --endpoints http://localhost:8000/v1 \
  --model pennywise-rloo --n 128 --concurrency 32
NEXT
