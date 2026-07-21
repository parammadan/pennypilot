#!/bin/bash
# SFT v2.1 + RLOO-50 session: retrain on the fixed demos -> full eval (the
# permission gate MUST be clean before RL — enforced in-job) -> 50-step RLOO.
#   sbatch --partition=gpu --gres=gpu:v100-sxm2:1 --cpus-per-task=8 --mem=64G \
#          --time=03:00:00 --job-name=v21-rloo \
#          --output=/scratch/madan.pa/pennypilot/v21_session_%j.log \
#          ~/pennywise-v100-infra/scripts/v21_session.sh
set -uo pipefail
cd "$HOME/pennywise-v100-infra"
source env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BASE=/scratch/madan.pa/pennypilot

echo "=== [1/4] tiny gate on the FIXED demo generator ==="
python scripts/run_sft_v2.py --demos 64 --max-steps 20 --batch-size 2 \
    --out-dir "$BASE/sft_v21_tiny" || exit 1
python - <<'PY' || exit 1
import json, math
rows = [json.loads(l) for l in open("/scratch/madan.pa/pennypilot/sft_v21_tiny/metrics.jsonl")]
assert all(math.isfinite(r["loss"]) for r in rows) and rows, "tiny gate failed"
assert sum(r["loss"] for r in rows[-5:])/5 < rows[0]["loss"], "loss not falling"
print("TINY GATE PASS")
PY

echo "=== [2/4] SFT v2.1 (1000 fixed demos) ==="
python scripts/run_sft_v2.py --demos 1000 --epochs 1 --batch-size 2 \
    --save-every 100 --out-dir "$BASE/sft_v21" || exit 1

echo "=== [3/4] full eval — permission gate must be CLEAN before RL ==="
python scripts/eval_v2.py --ckpt "$BASE/sft_v21/policy" --n 64 \
    --out "$BASE/eval/sft_v21.json"
RC=$?
if [ $RC -ne 0 ]; then
  echo "GATE NOT CLEAN (violations persist) — STOPPING before RL"; exit 1
fi

echo "=== [4/4] RLOO 50 observed steps from v2.1 (split A) ==="
python scripts/run_rl_v2.py --sft-adapter "$BASE/sft_v21/policy" --algo rloo \
    --steps 50 --k 8 --language es-en --n-must-haves 3 4 --valid-range 3 10 \
    --save-every 10 --out-dir "$BASE/rloo50_v2"

echo "=== V2.1 SESSION COMPLETE ==="
ls -la "$BASE/sft_v21" "$BASE/rloo50_v2" 2>/dev/null
