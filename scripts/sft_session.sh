#!/bin/bash
# Unattended SFT session (sbatch-able): tiny-config gate -> automated gate
# check -> full SFT -> held-out eval vs the measured base floor.
#
#   sbatch --partition=gpu-short --gres=gpu:v100-sxm2:1 --cpus-per-task=8 \
#          --mem=64G --time=02:00:00 --job-name=sft-v2 \
#          --output=/scratch/madan.pa/pennypilot/sft_session_%j.log \
#          ~/pennywise-v100-infra/scripts/sft_session.sh
#
# The tiny-config rule is enforced IN-JOB: the full run only starts if the
# 20-step tiny run's loss is finite and falling and the mask verified. Any
# failure stops the job right there with the evidence in the log.
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BASE=/scratch/madan.pa/pennypilot
mkdir -p "$BASE/eval"

echo "=== [1/3] tiny-config gate (20 steps, 64 demos) ==="
python scripts/run_sft_v2.py --demos 64 --max-steps 20 --batch-size 2 \
    --out-dir "$BASE/sft_tiny" || exit 1
python - <<'PY' || exit 1
import json, math
rows = [json.loads(l) for l in open("/scratch/madan.pa/pennypilot/sft_tiny/metrics.jsonl")]
assert len(rows) >= 20, f"tiny run too short: {len(rows)} steps"
assert all(math.isfinite(r["loss"]) for r in rows), "non-finite loss in tiny run"
assert all(r["supervised_tokens"] > 0 for r in rows), "empty supervised span"
first, last5 = rows[0]["loss"], sum(r["loss"] for r in rows[-5:]) / 5
assert last5 < first, f"loss did not fall: {first} -> {last5:.4f}"
print(f"TINY GATE PASS: loss {first} -> {last5:.4f} over {len(rows)} steps")
PY

echo "=== [2/3] full SFT (1000 demos, 1 epoch) ==="
python scripts/run_sft_v2.py --demos 1000 --epochs 1 --batch-size 2 \
    --save-every 100 --resume latest --out-dir "$BASE/sft_v2" || exit 1

echo "=== [3/3] held-out eval (en/es/es-en) vs base floor 12.5% ==="
python scripts/eval_v2.py --ckpt "$BASE/sft_v2/policy" --n 64 \
    --out "$BASE/eval/sft_v2.json"

echo "=== SFT SESSION COMPLETE ==="
ls -la "$BASE/sft_v2" "$BASE/eval"
