#!/bin/bash
# Block 2 items 10+11: tiny reuse gate -> H2 reuse arm -> n=128 eval ->
# friendly-phrasing retrain. All outputs quarantined; frozen ckpts read-only.
set -uo pipefail
cd "$HOME/pennywise-v100-infra"
source env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
BASE=/scratch/madan.pa/pennypilot
SFT="$BASE/sft_v21/policy"

echo "=== [1/4] tiny reuse gate (3 steps, k=4) ==="
python scripts/run_rl_v2.py --sft-adapter "$SFT" --algo grpo --reuse-epochs 4 \
  --steps 3 --k 4 --out-dir "$BASE/reuse_tiny" || exit 1
python - <<'PY' || exit 1
import json, math
rows=[json.loads(l) for l in open("/scratch/madan.pa/pennypilot/reuse_tiny/metrics.jsonl")]
assert len(rows)==3 and all(math.isfinite(r["kl_mean"]) for r in rows), rows
print("TINY REUSE GATE PASS:", [(r["step"], round(r["kl_mean"],5)) for r in rows])
PY

echo "=== [2/4] H2 reuse arm: GRPO + 4-epoch clipped reuse, 50 steps ==="
python scripts/run_rl_v2.py --sft-adapter "$SFT" --algo grpo --reuse-epochs 4 \
  --steps 50 --k 8 --language es-en --scenario-seed 3000 \
  --n-must-haves 3 4 --valid-range 3 10 --save-every 10 \
  --out-dir "$BASE/grpo50_reuse" || exit 1

echo "=== [3/4] win-condition eval (n=128, seed 5001) ==="
python scripts/eval_v2.py --ckpt "$BASE/grpo50_reuse/policy" --n 128 \
  --languages es-en --scenario-seed 5001 --n-must-haves 3 4 --valid-range 3 10 \
  --out "$BASE/compare/eval_grpo50_reuse.json" || true

echo "=== [4/4] friendly-phrasing retrain (quarantined sft_v23) ==="
python scripts/run_sft_v2.py --demos 64 --max-steps 20 --batch-size 2 \
  --out-dir "$BASE/sft_v23_tiny" || exit 1
python scripts/run_sft_v2.py --demos 1000 --epochs 1 --batch-size 2 \
  --save-every 200 --out-dir "$BASE/sft_v23" || exit 1

echo "=== BLOCK2 SESSION COMPLETE ==="
python - <<'PY'
import json
rows=[json.loads(l) for l in open("/scratch/madan.pa/pennypilot/grpo50_reuse/metrics.jsonl")]
kls=[r["kl_mean"] for r in rows]
print(f"reuse arm: {len(rows)} steps | KL final {kls[-1]:.5f} max {max(kls):.5f} | "
      f"reward last10 {sum(r['reward_mean'] for r in rows[-10:])/10:+.3f} | "
      f"viol steps {sum(r['violation_rate']>0 for r in rows)}")
r=json.load(open("/scratch/madan.pa/pennypilot/compare/eval_grpo50_reuse.json"))["reports"][0]
print(f"eval: success {r['task_success_rate']:.3f} viol {r['permission_violation_rate']:.3f}")
PY
