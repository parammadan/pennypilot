#!/bin/bash
# FINALE: SS11 (kernel gate) + SS13 (waterfall), staged across envs.
set -uo pipefail
cd "$HOME/pennywise-v100-infra"
source env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
ART=/scratch/madan.pa/pennypilot/artifacts
VPY=/scratch/madan.pa/venvs/vllm-073/bin/python
mkdir -p "$ART/ss11" "$ART/ss13"
P11="PG-math chain < 1% of the full iteration (update itself is ~4%) -> Triton kernel NOT justified; the documented no-go IS the artifact"
P13="final rung >= 4x faster than baseline; batched rollouts contribute nearly all of it; engine swap alone contributes ~nothing"

echo "=== SS11: update re-profile (kernel gate) ==="
python benchmarks/ss11_update_reprofile.py --sft-adapter /scratch/madan.pa/pennypilot/rloo50_v2/policy \
  --predicted "$P11" --out "$ART/ss11" 2>&1 | tee "$ART/ss11/run_${SLURM_JOB_ID:-x}.log" || exit 1

echo "=== SS13: merge adapter (trainer env) ==="
python benchmarks/ss13_waterfall.py --stage merge --out "$ART/ss13" || exit 1
echo "=== SS13: HF baseline rollouts ==="
python benchmarks/ss13_waterfall.py --stage rollout_hf --out "$ART/ss13" || exit 1
echo "=== SS13: vLLM rollouts (merged, venv, clean LD path) ==="
env -u LD_LIBRARY_PATH VLLM_USE_V1=0 HF_HOME=/scratch/madan.pa/hf_cache HF_HUB_OFFLINE=1 \
  $VPY benchmarks/ss13_waterfall.py --stage rollout_vllm --out "$ART/ss13" || exit 1
echo "=== SS13: update timings ==="
python benchmarks/ss13_waterfall.py --stage update --out "$ART/ss13" || exit 1
echo "=== SS13: assemble waterfall ==="
python benchmarks/ss13_waterfall.py --stage assemble --predicted "$P13" --out "$ART/ss13" || exit 1

echo "=== FINALE SESSION COMPLETE ==="
ls "$ART/ss11" "$ART/ss13"
