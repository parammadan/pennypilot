# PennyData + PennyMart — live demo runbook

The show: a shopping agent (RL post-trained 7B) serving real conversations in
a browser store, every interaction streaming through Kafka into an open data
platform — analytics live on one screen, cold storage filling in S3 — and the
platform exporting validated training data back out. The flywheel, end to end.

## 0. Preflight (5 min before)

```bash
brew services start kafka                  # broker (KRaft, localhost:9092)
cd ~/pennywise-v100-infra
# GPU: serve BOTH models on one V100 (B3 = the shipped 7B)
ssh explorer 'cd ~/pennywise-v100-infra && D=/scratch/madan.pa/pennypilot; \
  sbatch --partition=sharing --gres=gpu:v100:1 --cpus-per-task=4 --mem=48G \
  --time=01:00:00 scripts/serve_duo.sh $D/rl7b_B3/policy'
# when the log prints the node (usually ~3 min):
ssh -f -N -o ControlMaster=no -o ControlPath=none \
    -L 8765:<NODE>:8765 -L 8766:<NODE>:8766 explorer
curl -s --noproxy localhost http://localhost:8766/health   # 7B-B3 ready?
```

## 1. Start the platform (hot path + console)

```bash
.venv/bin/python scripts/run_platform.py --root ~/pennydata \
    --port 8770 --kafka localhost:9092 --consumers 2 &
open http://localhost:8770                 # the PennyData console
```

## 2. Background production traffic (live, paced)

```bash
.venv/bin/python scripts/synth_traffic.py --n 100000 --rate 3 \
    --kafka localhost:9092 &               # ~3 episodes/sec, runs all demo
```
Console tiles/funnel now move on their own — that's "live ingestion."

## 3. Cold path (S3 archival)

```bash
.venv/bin/python scripts/run_s3_sink.py \
    --bucket pennydata-771965334314-us-east-2 &
# show it later:  aws s3 ls s3://pennydata-771965334314-us-east-2/pennydata/events/ --recursive | tail
```

## 4. The A/B beat (two stores, two brains)

```bash
# window 1 — the 1.5B specialist (shops great, FORGOT how to chat):
.venv/bin/python scripts/demo_human.py --label 1.5b \
    --kafka localhost:9092 --save-dir demos/live_sessions
# window 2 — the 7B rehearsal+B3 (chats AND shops AND says "can't do that"):
.venv/bin/python scripts/demo_human.py --chat --chat-min \
    --policy-url http://localhost:8766 --label 7b-B3 \
    --kafka localhost:9092 --save-dir demos/live_sessions
```
Script for each window (same inputs, watch the contrast):
1. `Hola, necesito una laptop, ¿me ayudas?` → give budget/brand/RAM when asked
2. Mid-flow: `what's the capital of France?`   ← 1.5B emits a shopping action;
   7B answers, then returns to task
3. `minimum $2500 please`                      ← 7B explains floors are
   unsupported and asks for a max (the B-loop fix, live)
4. `do you sell headphones?`                   ← 7B: "laptops only", redirects
5. Permission modal: hit **Not yet** once (denial recovery), then Approve
6. Click 👍/👎 on replies — votes appear in the console feed instantly

## 5. Close the flywheel (on the console)

- Self-service SQL: `SELECT label, COUNT(*), SUM(violation) FROM episodes GROUP BY label`
- **Export SFT dataset** button → validated training sequences + the report
  (hard-example candidates = your thumbs-downs)
- Line to say: "this export is the same format the SFT trainer consumes —
  the loop you just fed is the loop that trained the model answering you."

## 6. Teardown

```bash
kill %1 %2 %3 2>/dev/null                  # platform, traffic, sink
ssh explorer 'scancel $(squeue -u madan.pa -h -o %i)'   # free the V100
pkill -f "ssh.*-L 8765"                    # tunnel
brew services stop kafka                   # optional
```

## Talking points if probed

- Kafka: keyed by session (per-conversation order), acks=all, at-least-once +
  idempotent store = effectively-once; replay = fresh consumer group (offsets
  belong to the group — hit this for real, see docs CHALLENGES).
- Scaling limit we measured: 4 consumers serialize on one SQLite writer —
  the "why real platforms shard sinks / use a warehouse" answer.
- Orchestration: the training pipelines are Slurm DAGs (`--dependency=afterok`
  chains with fail-fast probe gates) — see docs PROGRESS 2026-07-24.
- Deliberately NOT run: Spark/Flink (nothing honest to compute at this scale),
  MSK (cost); both described as the deployment path, not faked.
