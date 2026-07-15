# Phase 2 runbook — run on the NEU Explorer login node

You are Claude Code on the **NEU Explorer SLURM login node** (login `madan.pa`).
GPU time is reserved. The repo is `github.com/parammadan/pennywise-v100-infra`.

**Read `ARCHITECTURE.md` and `PHASE2_NOTES.md` first** — they define the blockers
(B1–B4) and the ordering rule. Do the steps **in order**. **Profiling (Step 3)
comes before setting any config numbers** — everything depends on the measured
max sequence length. **STOP after Step 4.** Keep each step tested and committed.
Many short *observed* runs, never one long blind batch. Log measured numbers
only.

## 0. Setup
```
cd ~ && git clone https://github.com/parammadan/pennywise-v100-infra.git \
     || (cd ~/pennywise-v100-infra && git pull)
cd ~/pennywise-v100-infra && source env.sh      # cuda module + conda env `shoprl`
pip install -e .                                # pydantic/pyyaml
python -c "import torch,transformers,peft" || pip install torch transformers peft
python -m pytest -q                             # 24 Phase-1 tests should pass
```
Note the SLURM **max walltime** (drives checkpoint frequency). No AWS billing
here — that's Phase 3 (A10G). The credential scan is already clean.

## A. Discover the cluster's real values (do NOT assume names)
```
sinfo -o "%P %G %l %N"
scontrol show partition | grep -E "PartitionName|MaxTime|State"
module avail 2>&1 | grep -i cuda
```
Identify: the **GPU partition name**, the **exact V100 GRES string** (e.g.
`gpu:v100:1` or `gpu:v100-sxm2:1`), the **max walltime**, the **CUDA module**.
Use THESE real values below.

## B. Grab an interactive V100 session
```
srun --partition=<PARTITION> --gres=<V100_GRES> --cpus-per-task=8 --mem=64G \
     --time=4:00:00 --pty /bin/bash
module load <cuda-module>
nvidia-smi          # CONFIRM V100, ~32GB — report the exact MiB
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Step 1 — Full fine-tune trainer path (blocker B1)
`src/shoprl/grpo/trainer.py::build_policy` **always** applies LoRA. Add a full-FT
path, selected by config:
- `src/shoprl/config.py` `TrainingConfig`: add `finetune: Literal["full","lora"] = "lora"` (default preserves current behaviour).
- Full-FT branch: load in **fp16** (Volta = no bf16), train **all** params (no
  `get_peft_model`).
- **KL reference:** with full-FT there is no adapter to disable, so the frozen
  base is no longer a free reference. Load a **separate frozen copy** of the base
  (`.eval()`, `no_grad`) as the KL reference and use it where the code currently
  calls `model.disable_adapter()`. (1.5B fp16 ≈ 3 GB — fits.) Keep the LoRA path
  intact for the fallback.
- Keep gradient checkpointing wired for the full-FT memory.
- Commit once it imports and a 1-step sanity (tiny model) runs on the GPU.

## Step 2 — 1.5B instruct config with PLACEHOLDER numbers
Add `configs/qwen2_5_1_5b_instruct_v100.yaml`:
- `model.name: Qwen/Qwen2.5-1.5B-Instruct`, `model.dtype: float16`, `finetune: full`.
- Leave `max_new_tokens`, `num_samples`, `max_turns`, batch as **placeholders**
  with `# SET AFTER STEP-3 PROFILING`. Do NOT guess them. Commit.

## Step 3 — VRAM / max-sequence-length profiling (BEFORE any config numbers)
Volta has **no FlashAttention** → attention memory is **quadratic in sequence
length**. This is the gate.
- Write `scripts/profile_v100.py`: load Qwen2.5-1.5B-Instruct full-FT on the
  V100, run forward+backward at increasing seq lengths (512, 768, 1024, 1536,
  2048) × batch/group sizes, with grad-checkpointing on and off. Record
  `torch.cuda.max_memory_allocated` per config; mark OK/OOM under a ~30 GB cap
  (leave headroom). Save `runs/profile_v100.json` + print a table.
- **Decision:** if full-FT does not fit the multi-turn sequence length you need
  (estimate = `max_turns` × per-turn tokens), **switch to LoRA** (`finetune:
  lora`) + gradient checkpointing. That is the correct **measured** outcome, not
  a failure — record the reason.
- From the curve, set the REAL numbers in the 1.5B config: max viable seq len →
  `max_turns` (env) + `max_new_tokens`; batch/group → `num_samples` (target k=8
  if it fits). Commit the script + `runs/profile_v100.json` + the filled config.

## Step 4 — Bounded SFT warmup
- Generate ~1k demos with `data/sft.py::generate_sft_dialogues` (built + tested
  on CPU).
- Write the SFT formatter + trainer (supervised, **not** RL): render each `Demo`
  into the model's chat template, **mask the loss to the AGENT turns only**, SFT
  the 1.5B for a **bounded**, observed number of steps (small — this is a
  warmup).
- **Success signal:** after SFT, run the model on a few held-out openers, parse
  its actions with the grammar (`PennyEnv._parse` / the regex in
  `tests/test_sft.py`), and report the **fraction of well-formed actions**. That
  is the Step-4 signal (not reward).
- Checkpoint the SFT model; commit the SFT code + a short `SFT_RESULTS.md`.

## STOP after Step 4 — report
1. Confirmed GPU (exact VRAM MiB from `nvidia-smi`).
2. Profiled **max seq length**, chosen **max_turns**, chosen **batch/group**.
3. **Full-FT vs LoRA** decision, with the measured reason.
4. Whether SFT produces **valid grammar** (the well-formed-action %).

The RLOO loop is the **next** round: 50-step observed iterations; confirm reward
moves and KL stays controlled before scaling. Do not launch a blind batch.
