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

# ---------------------------------------------------------------------------
# SERVE — KNOWN ISSUE (2026-07-16, first A10G run):
# The DLAMI's default `pip install vllm` pulled vLLM 0.25.1, whose v1 engine
# HARD-REQUIRES flashinfer, and flashinfer's runtime JIT sampler build
# (get_sampling_module().build_and_load()) FAILED on this box -> "Engine core
# initialization failed" every launch. Removing flashinfer then broke load_model
# (v1 engine imports it there too). --enforce-eager and VLLM_USE_FLASHINFER_SAMPLER=0
# did NOT help. So serving never came up; instance was terminated.
#
# FIX FOR NEXT ATTEMPT (pin a tested stack instead of latest):
#   pip install "vllm==0.6.6"        # stable v0-engine, no flashinfer JIT req
#     (may need matching torch; if so use a vLLM-pinned image/venv)
#   OR provide a prebuilt flashinfer wheel matching vllm so no JIT build runs
#   OR VLLM_USE_V1=0 (if the pinned vllm still supports the v0 engine)
# Then, once "Application startup complete" is in vllm.log:
#   python3 scripts/rollout_vllm.py --endpoints http://localhost:8000/v1 \
#     --model pennywise-rloo --n 128 --concurrency 32
# ---------------------------------------------------------------------------
cat <<'NEXT'
--- serve (after pinning a working vllm — see notes above) ---
nohup vllm serve ~/pennywise/ckpt --served-model-name pennywise-rloo \
  --dtype bfloat16 --max-model-len 2048 --gpu-memory-utilization 0.9 \
  < /dev/null > ~/pennywise/vllm.log 2>&1 &
python3 scripts/rollout_vllm.py --endpoints http://localhost:8000/v1 \
  --model pennywise-rloo --n 128 --concurrency 32
NEXT
