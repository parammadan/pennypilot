#!/bin/bash
# Source this at the start of every session on an Explorer GPU node:
#   source env.sh
module load nodejs miniconda3 cuda/12.3.0 cuDNN git 2>/dev/null
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate shoprl

# Prevent ~/.local/lib/python3.12/site-packages (a prior `pip install --user`)
# from shadowing packages installed into this conda env -- it sits earlier on
# sys.path than the env's own site-packages by default.
export PYTHONNOUSERSITE=1

# Keep HF model/dataset downloads off the home quota.
export HF_HOME=/scratch/madan.pa/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
