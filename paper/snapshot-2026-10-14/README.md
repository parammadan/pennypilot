# Manuscript snapshot, 14 October 2026

Frozen copy of the paper for the Vanderbilt PhD application. Do not edit;
the working copy for the conference version is `paper/rehearsal/`.

- `main.pdf`: compiled with `tectonic -X compile main.tex`
- Git tag: `snapshot-2026-10-14`

## Sources for every number
- Tables 1–2: `paper/results/results.csv` (`scripts/paper/make_table.py`)
  and `paper/results/stats.json` (`scripts/paper/stats.py`)
- Table 3 and Section 6: `paper/results/behavior.json`
  (`scripts/paper/behavior_table.py paper/results/raw`), aggregated from
  `paper/results/raw/*/behavior/behavior.json`. That covers 13 checkpoints:
  the base model, plus specialist and rehearsal25 after SFT and after RL,
  seeds 1–3. Each eval has its own `manifest.json` (git sha, SLURM job id,
  adapter path).

## Included
- Everything in the previous draft (commit bf04416): 3 seeds, 2 arms, SFT and RL
- New: off-task tool rate under 4 system prompts, TriviaQA/MGSM correctness
  with and without the system prompt, and the LoRA-scale sweep. All of it is
  complete for every seed of both arms.

## Excluded
- On-task first-turn tool rate. All 128 on-task openers are the same
  string, so it is one prompt repeated 128 times, not 128 measurements.
- Every in-progress arm: rehearsal10, rehearsal50, human25, klbase0.1,
  klbase1, seeds 4–5, and Mistral. Some of their seeds have finished, but no
  arm is complete, so none appear here. See `paper/ONGOING.md`.
