#!/bin/bash
# THE PRE-REGISTERED COMPARISON (HYPOTHESES.md + 2026-07-21 amendment):
# GRPO at equal rollout budget vs the completed RLOO-50, then the n=128
# win-condition eval over three arms. Chained after the SS0-SS2 session.
#   sbatch --dependency=afterok:<ss_jobid> --partition=gpu \
#          --gres=gpu:v100-sxm2:1 --cpus-per-task=8 --mem=64G --time=04:00:00 \
#          --job-name=grpo-compare \
#          --output=/scratch/madan.pa/pennypilot/compare_%j.log \
#          ~/pennywise-v100-infra/scripts/comparison_session.sh
set -uo pipefail
cd "$HOME/pennywise-v100-infra"
source env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
BASE=/scratch/madan.pa/pennypilot
SFT="$BASE/sft_v21/policy"
mkdir -p "$BASE/compare"

echo "=== [1/2] GRPO-50 — identical config to RLOO-50 except the advantage rule ==="
python scripts/run_rl_v2.py --sft-adapter "$SFT" --algo grpo \
    --steps 50 --k 8 --language es-en --scenario-seed 3000 \
    --n-must-haves 3 4 --valid-range 3 10 \
    --save-every 10 --out-dir "$BASE/grpo50_v2" || exit 1

echo "=== [2/2] win-condition eval: n=128, seed 5001, RL-split difficulty, es-en ==="
for ARM in sft_v21 rloo50_v2 grpo50_v2; do
  python scripts/eval_v2.py --ckpt "$BASE/$ARM/policy" --n 128 --languages es-en \
      --scenario-seed 5001 --n-must-haves 3 4 --valid-range 3 10 \
      --out "$BASE/compare/eval_$ARM.json" || true   # violations are DATA here
done

python - <<'PY'
import json
print("\n=== PRE-REGISTERED COMPARISON — RAW TABLE (interpretation in docs) ===")
print(f"{'arm':<12} {'success':>8} {'viol':>6} {'value':>7} {'invalid':>8}")
for arm in ("sft_v21", "rloo50_v2", "grpo50_v2"):
    r = json.load(open(f"/scratch/madan.pa/pennypilot/compare/eval_{arm}.json"))["reports"][0]
    print(f"{arm:<12} {r['task_success_rate']:>8.3f} {r['permission_violation_rate']:>6.3f} "
          f"{r['mean_value_quality']:>7.3f} {r['invalid_action_rate']:>8.4f}")
for run in ("rloo50_v2", "grpo50_v2"):
    rows=[json.loads(l) for l in open(f"/scratch/madan.pa/pennypilot/{run}/metrics.jsonl")]
    kls=[m['kl_mean'] for m in rows]
    print(f"{run}: KL final {kls[-1]:.5f} max {max(kls):.5f} | "
          f"viol steps {sum(m['violation_rate']>0 for m in rows)}/{len(rows)} | "
          f"reward last10 {sum(m['reward_mean'] for m in rows[-10:])/10:+.3f}")
PY
echo "=== COMPARISON SESSION COMPLETE ==="
