# Ongoing experiments (not in the 2026-10-14 snapshot)

Results land in /scratch/madan.pa/paper/eval/<arm>__s<seed>__paper.
Archived copies are in ~/paper_archive/2026-10-14 on the cluster and
~/pennypilot_archive/2026-10-14 locally. Adapters are mirrored to
~/paper_ckpts on the cluster.

| Arm | What | State on 2026-10-14 |
|---|---|---|
| rehearsal10 / rehearsal50 | dose curve | partial; see eval dirs |
| human25 | human-written Dolly answers, same prompts | partial |
| specialist / rehearsal25 s4–s5 | more seeds | specialist SFT s4/s5 evaluated; rest queued |
| klbase0.1 / klbase1 | KL-to-base SFT baseline | queued |
| mistral-* | second model family | queued; rehearsal set ready (660 rows) |
| behavior evals for the new arms | eval A | not yet submitted |

## Known issue to fix before the conference version
- `eval_behavior.py` on-task set: `env.reset()` returns the same opener for
  every scenario (128 copies of one string). Either drop the metric or build
  varied on-task openers. The off-task, correctness and sweep metrics are
  unaffected.
