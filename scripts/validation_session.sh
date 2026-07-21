#!/bin/bash
# VALIDATION SESSION (first GPU request of PennyPilot) — 2h interactive V100.
#
# Get the allocation, then run this from the repo root INSIDE the srun shell:
#   srun --partition=gpu-short --gres=gpu:v100-sxm2:1 --cpus-per-task=8 \
#        --mem=64G --time=02:00:00 --pty bash
#   cd ~/pennywise-v100-infra && bash scripts/validation_session.sh
#
# PREP (login node, BEFORE the allocation — compute nodes have no egress):
#   1. huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct   (HF_HOME on scratch)
#   2. one venv per vLLM candidate (see docs repo RUN_ON_SLURM.md §9):
#        python -m venv /scratch/madan.pa/venvs/vllm-073 && \
#          /scratch/madan.pa/venvs/vllm-073/bin/pip install vllm==0.7.3
#        python -m venv /scratch/madan.pa/venvs/vllm-092 && \
#          /scratch/madan.pa/venvs/vllm-092/bin/pip install vllm==0.9.2
#      (each venv also needs: pip install -e ~/pennywise-v100-infra — no, the
#       smoke script is stdlib+vllm only; just copy scripts/vllm_smoke.py path)
#   3. source env.sh  (modules, conda env shoprl, HF_HOME, HF_HUB_OFFLINE=1)
set -uo pipefail
[ -f env.sh ] && source env.sh    # modules, conda env, HF_HOME
# Compute nodes have no egress: force cache-only loads so nothing stalls
# probing huggingface.co (the egress CHECK below still tests the real thing).
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
GATE_DIR=${GATE_DIR:-/scratch/madan.pa/pennypilot/gate}
mkdir -p "$GATE_DIR"
LOG="$GATE_DIR/session_$(date +%Y%m%d_%H%M).log"
exec > >(tee "$LOG") 2>&1
echo "=== PennyPilot validation session $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo; echo "--- [1/5] compute-node egress check ---"
python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=5); print('egress: OK')" \
  || echo "egress: NONE (expected) -> stay on HF_HUB_OFFLINE=1 + WANDB_MODE=offline"

echo; echo "--- [2/5] precision micro-benchmark (fp16+GradScaler vs bf16-emulated) ---"
python scripts/bench_precision.py --steps 30 --method lora \
  --out "$GATE_DIR/precision.json"

echo; echo "--- [3/5] vLLM candidate ladder (first PASS wins, then STOP) ---"
for V in /scratch/madan.pa/venvs/vllm-*; do
  [ -d "$V" ] || continue
  [ -f "$V.READY" ] || { echo ">>> skipping $V (install not finished)"; continue; }
  echo ">>> candidate venv: $V"
  if VLLM_USE_V1=0 "$V/bin/python" scripts/vllm_smoke.py \
       --model Qwen/Qwen2.5-1.5B-Instruct --gpu-mem-util 0.45 \
       --out "$GATE_DIR/vllm_smoke_$(basename "$V").json"; then
    echo ">>> WINNER: $V — record pin in docs repo RUN_ON_SLURM.md §9 and FREEZE"
    break
  fi
done

echo; echo "--- [4/5] SCENARIO HARDNESS GATE (base-model success, band 10-40%) ---"
python scripts/base_success.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --n 64 --language es-en --out "$GATE_DIR/base_success.json"

echo; echo "--- [5/5] wrap-up ---"
echo "Artifacts in $GATE_DIR:"; ls -la "$GATE_DIR"
cat <<'EON'
Remaining MANUAL items for this session:
  - SIGTERM/requeue drill: submit the sbatch template with a dummy 5-min loop,
    then: scancel --signal=TERM <jobid>; verify checkpoint+exit-0+requeue.
  - When done: EXIT the srun shell immediately — release the allocation.
Paste the three JSONs back to Claude for the dated PROGRESS.md entry
(wall-time used vs requested included).
EON
