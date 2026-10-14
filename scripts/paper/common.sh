# Sourced by the paper job scripts. Scratch is PURGED periodically (2026-10:
# every July checkpoint lost), so finished adapters are mirrored to $HOME.
export HF_HOME=/scratch/madan.pa/.hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd "$HOME/pennywise-v100-infra"
PY=/home/madan.pa/.conda/envs/shoprl/bin/python
SCR=/scratch/madan.pa/paper
KEEP=$HOME/paper_ckpts                     # durable mirror (adapters are small)
M7=Qwen/Qwen2.5-7B-Instruct
REH=${REH:-$SCR/rehearsal_dolly.jsonl}   # override for the human-written control
mirror() {  # mirror <run-dir> : copy adapter + manifest + metrics, no optimizer state
  local src="$1" dst="$KEEP/$(basename "$(dirname "$1")")/$(basename "$1")"
  mkdir -p "$dst"
  rsync -a --exclude 'step-*' --exclude 'train_state.pt' "$src/" "$dst/"
  echo "[mirror] $src -> $dst ($(du -sh "$dst" | cut -f1))"
}
