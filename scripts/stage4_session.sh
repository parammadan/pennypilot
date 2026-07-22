#!/bin/bash
# Stage-4 session (chained after the comparison): WebShop programmatic eval of
# the trained policy on unseen products + trained-policy demo-transcript
# capture for Stage 5's replay.
#   sbatch --dependency=afterok:<compare_jobid> --partition=gpu-short,gpu \
#          --gres=gpu:v100-sxm2:1 --cpus-per-task=8 --mem=64G --time=01:00:00 \
#          --job-name=stage4 \
#          --output=/scratch/madan.pa/pennypilot/stage4_%j.log \
#          ~/pennywise-v100-infra/scripts/stage4_session.sh
set -uo pipefail
cd "$HOME/pennywise-v100-infra"
source env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
BASE=/scratch/madan.pa/pennypilot
POLICY="$BASE/rloo50_v2/policy"
mkdir -p "$BASE/webshop" "$BASE/demo_bundles"

echo "=== [1/2] Stage 4: WebShop eval (unseen products, EN/ES/Spanglish) ==="
python scripts/webshop_eval.py --ckpt "$POLICY" \
    --out "$BASE/webshop/eval_rloo50.json" || exit 1

echo "=== [2/2] Stage-5 prep: trained-policy demo transcripts ==="
for LANG in en es es-en; do
  python scripts/make_demo_transcript.py --ckpt "$POLICY" --language "$LANG" \
      --scenario-index 1 --out "$BASE/demo_bundles/trained_$LANG.json"
done
ls -la "$BASE/demo_bundles"
echo "=== STAGE 4 SESSION COMPLETE ==="
