#!/bin/bash
# Unattended RL bring-up session (sbatch-able):
#   [1] Spanish permission-violation capture (blocker from the SFT eval)
#   [2] RL-split hardness calibration (two candidate splits; pick ~40-60%)
#   [3] tiny-config RLOO on the chosen split (10 observed steps)
#
#   sbatch --partition=gpu-short --gres=gpu:v100-sxm2:1 --cpus-per-task=8 \
#          --mem=64G --time=02:00:00 --job-name=rl-bringup \
#          --output=/scratch/madan.pa/pennypilot/rl_session_%j.log \
#          ~/pennywise-v100-infra/scripts/rl_session.sh
set -uo pipefail
cd "$HOME/pennywise-v100-infra"
source env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BASE=/scratch/madan.pa/pennypilot
SFT="$BASE/sft_v2/policy"
mkdir -p "$BASE/eval" "$BASE/calib"

echo "=== [1/3] Spanish violation capture (greedy es split, transcripts on) ==="
python scripts/eval_v2.py --ckpt "$SFT" --n 64 --languages es \
    --out "$BASE/eval/es_recheck.json" || true   # nonzero exit on violations is EXPECTED
ls "$BASE/eval/violations/" 2>/dev/null || echo "(no violation reproduced — note for diagnosis)"

echo "=== [2/3] RL-split calibration (target: success 0.40-0.60) ==="
python scripts/eval_v2.py --ckpt "$SFT" --n 32 --languages es-en \
    --scenario-seed 3000 --n-must-haves 3 4 --valid-range 3 10 \
    --out "$BASE/calib/splitA.json" || true
python scripts/eval_v2.py --ckpt "$SFT" --n 32 --languages es-en \
    --scenario-seed 3000 --n-must-haves 4 4 --valid-range 2 6 \
    --out "$BASE/calib/splitB.json" || true
python - <<'PY'
import json
best, best_d = None, 9
for name, mh, vr in (("A", [3, 4], [3, 10]), ("B", [4, 4], [2, 6])):
    r = json.load(open(f"/scratch/madan.pa/pennypilot/calib/split{name}.json"))
    s = r["reports"][0]["task_success_rate"]
    d = abs(s - 0.5)
    print(f"split {name}: success={s:.2f} (must-haves {mh}, valid {vr})")
    if d < best_d:
        best, best_d, best_mh, best_vr = name, d, mh, vr
choice = {"split": best, "n_must_haves": best_mh, "valid_range": best_vr}
json.dump(choice, open("/scratch/madan.pa/pennypilot/calib/choice.json", "w"))
print("CHOSEN:", choice)
PY

echo "=== [3/3] tiny RLOO (10 observed steps on the chosen split) ==="
MH=$(python -c "import json;c=json.load(open('$BASE/calib/choice.json'));print(*c['n_must_haves'])")
VR=$(python -c "import json;c=json.load(open('$BASE/calib/choice.json'));print(*c['valid_range'])")
python scripts/run_rl_v2.py --sft-adapter "$SFT" --algo rloo \
    --steps 10 --k 8 --language es-en \
    --n-must-haves $MH --valid-range $VR \
    --out-dir "$BASE/rl_tiny"

echo "=== RL BRING-UP SESSION COMPLETE ==="
ls -la "$BASE/eval" "$BASE/calib" "$BASE/rl_tiny" 2>/dev/null
