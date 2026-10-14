#!/bin/bash
# CPU-only feeder: submits queued chains as gpu-QOS slots free up, then exits.
# Queue file lines: "<arm> <seed> <rehearsal_n> [rehearsal_file]". Runs as a 1-CPU 'short' job,
# so it needs no laptop/login session. Polls every 5 min.
set -uo pipefail
Q=/scratch/madan.pa/paper/queue.txt
D=$HOME/pennywise-v100-infra/scripts/paper
while [ -s "$Q" ]; do
  n=$(squeue -u "$USER" -p gpu -h | wc -l)
  if [ "$n" -le 6 ]; then
    line=$(head -1 "$Q")
    set -- $line
    if REH=${4:-${REH:-}} bash $D/submit_arm2.sh "$1" "$2" "$3"; then sed -i 1d "$Q"; fi
  fi
  sleep 300
done
echo "[feeder] queue empty, exiting $(date)"
