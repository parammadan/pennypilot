"""SS13 — THE WATERFALL: one RL iteration, rung by rung (staged across envs).

Stages (phases timed in their native envs, iteration = rollout + update):
  --stage merge        (trainer env)  merge the LoRA adapter -> plain weights
                       (the merge-at-sync design; vLLM-LoRA is Volta-broken)
  --stage rollout_hf   (trainer env)  k=8 sequential HF rollouts (baseline)
  --stage rollout_vllm (vLLM venv)    merged weights: sequential + batched-8
  --stage update       (trainer env)  update on recorded trajs, ckpt on & off
  --stage assemble     (trainer env)  waterfall chart + manifest

All stages write JSON into --out; assemble sums phases per rung:
  A baseline      = HF sequential rollout + update(ckpt-on)
  B +vLLM engine  = vLLM sequential   + update(ckpt-on)
  C +batched-8    = vLLM batched-8    + update(ckpt-on)
  D +ckpt-off     = vLLM batched-8    + update(ckpt-off)
"""
from __future__ import annotations

import argparse
import json
import os
import time

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _scenario():
    from shoprl.data.catalog import catalog_index, generate_catalog
    from shoprl.env.scenario import generate_hard_scenarios
    catalog = generate_catalog(n=300, seed=0)
    idx = catalog_index(catalog)
    scen = generate_hard_scenarios(catalog, n=1, seed=3000,
                                   n_must_haves=(3, 4),
                                   valid_target_range=(3, 10))[0]
    return catalog, idx, scen


def stage_merge(args) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, args.sft_adapter).merge_and_unload()
    merged.save_pretrained(args.merged_dir)
    # Deliberately NO tokenizer in merged_dir: this env's transformers 5.x
    # writes configs the vLLM venv's 4.49 cannot parse (CHALLENGES #23) — the
    # engine gets tokenizer=MODEL (base, old-format cache) instead.
    print(f"[SS13:merge] merged policy -> {args.merged_dir}")


def stage_rollout_hf(args) -> None:
    import torch
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2
    from shoprl.env.catalog_env import SyntheticCatalogEnvironment
    from shoprl.profiling.bench_common import load_hf_policy
    from shoprl.train.algo import rollout_v2
    catalog, idx, scen = _scenario()
    policy, tok = load_hf_policy(MODEL, args.sft_adapter)

    @torch.no_grad()
    def agent_fn(messages):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      tokenize=True, return_dict=True,
                                      return_tensors="pt")
        ids = enc["input_ids"].to("cuda")
        out = policy.generate(ids, max_new_tokens=64, do_sample=True,
                              temperature=1.0, top_p=0.95,
                              pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    t0 = time.time()
    trajs = []
    for _ in range(args.k):
        env = SyntheticCatalogEnvironment(catalog, scen, idx=idx, language="es-en")
        trajs.append(rollout_v2(agent_fn, env, SYSTEM_PROMPT_V2))
    wall = round(time.time() - t0, 1)
    json.dump({"rollout_hf_seconds": wall,
               "messages": [t.messages for t in trajs],
               "rewards": [t.reward for t in trajs]},
              open(os.path.join(args.out, "rollout_hf.json"), "w"))
    print(f"[SS13:rollout_hf] k={args.k} in {wall}s")


def stage_rollout_vllm(args) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2
    from shoprl.env.catalog_env import SyntheticCatalogEnvironment
    from shoprl.train.batch_rollout import batched_rollouts
    catalog, idx, scen = _scenario()
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=args.merged_dir, tokenizer=MODEL, dtype="float16",
              gpu_memory_utilization=0.45, enforce_eager=True)
    sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=64)

    def gen_batch(message_lists):
        prompts = [tok.apply_chat_template(m, tokenize=False,
                                           add_generation_prompt=True)
                   for m in message_lists]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outs]

    results = {}
    trajs_out = None
    for mode, batched in (("sequential", False), ("batched8", True)):
        t0 = time.time()
        if batched:
            envs = [SyntheticCatalogEnvironment(catalog, scen, idx=idx,
                                                language="es-en")
                    for _ in range(args.k)]
            trajs = batched_rollouts(gen_batch, envs, SYSTEM_PROMPT_V2)
        else:
            trajs = []
            for _ in range(args.k):
                env = SyntheticCatalogEnvironment(catalog, scen, idx=idx,
                                                  language="es-en")
                trajs.append(batched_rollouts(gen_batch, [env],
                                              SYSTEM_PROMPT_V2)[0])
        results[mode] = round(time.time() - t0, 1)
        trajs_out = trajs
        print(f"[SS13:rollout_vllm] {mode}: {results[mode]}s")
    json.dump({"rollout_vllm_sequential_seconds": results["sequential"],
               "rollout_vllm_batched8_seconds": results["batched8"],
               "messages": [t.messages for t in trajs_out],
               "rewards": [t.reward for t in trajs_out]},
              open(os.path.join(args.out, "rollout_vllm.json"), "w"))


def stage_update(args) -> None:
    import torch
    from shoprl.profiling.bench_common import load_hf_policy
    from shoprl.train.algo import RLOO
    from shoprl.train.rloo import _k3_kl, _sequence_and_mask, _token_logprobs
    data = json.load(open(os.path.join(args.out, "rollout_vllm.json")))
    messages, rewards = data["messages"], data["rewards"]
    advs = RLOO().advantages(rewards)
    policy, tok = load_hf_policy(MODEL, args.sft_adapter)
    reference, _ = load_hf_policy(MODEL, args.sft_adapter)
    for p in reference.parameters():
        p.requires_grad_(False)
    for n, p in policy.named_parameters():
        if "lora" in n.lower():
            p.requires_grad_(True)
            p.data = p.data.float()
    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad), lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    out = {}
    for label, ckpt in (("ckpt_on", True), ("ckpt_off", False)):
        policy.train()
        policy.config.use_cache = False
        (policy.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
         if ckpt else policy.gradient_checkpointing_disable())
        torch.cuda.synchronize()
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for msgs, A in zip(messages, advs):
            ids, mask = _sequence_and_mask(tok, msgs, 2048)
            asst = torch.tensor(mask[1:], device="cuda", dtype=torch.bool)
            if asst.sum() == 0:
                continue
            with torch.autocast("cuda", dtype=torch.float16):
                lp_pol = _token_logprobs(policy, ids, "cuda")
                with torch.no_grad():
                    lp_ref = _token_logprobs(reference, ids, "cuda")
            loss = (-A * lp_pol[asst].mean()
                    + 0.04 * _k3_kl(lp_pol[asst], lp_ref[asst]).mean()) / len(advs)
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        out[f"update_{label}_seconds"] = round(time.time() - t0, 1)
        print(f"[SS13:update] {label}: {out[f'update_{label}_seconds']}s")
    json.dump(out, open(os.path.join(args.out, "update.json"), "w"))


def stage_assemble(args) -> None:
    from shoprl.profiling import require_prediction, write_manifest
    predicted = require_prediction(args.predicted)
    hf = json.load(open(os.path.join(args.out, "rollout_hf.json")))
    vl = json.load(open(os.path.join(args.out, "rollout_vllm.json")))
    up = json.load(open(os.path.join(args.out, "update.json")))
    rungs = [
        ("A baseline\n(HF seq + ckpt-on)",
         hf["rollout_hf_seconds"] + up["update_ckpt_on_seconds"]),
        ("B +vLLM engine\n(seq, merged)",
         vl["rollout_vllm_sequential_seconds"] + up["update_ckpt_on_seconds"]),
        ("C +batched-8\nrollouts",
         vl["rollout_vllm_batched8_seconds"] + up["update_ckpt_on_seconds"]),
        ("D +ckpt-off\nupdate",
         vl["rollout_vllm_batched8_seconds"] + up["update_ckpt_off_seconds"]),
    ]
    speedup = round(rungs[0][1] / rungs[-1][1], 2)
    measured = " → ".join(f"{v:.1f}s" for _, v in rungs) + f" ({speedup}× total)"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor="#fcfcfb")
    labels = [r[0] for r in rungs]
    vals = [r[1] for r in rungs]
    bars = ax.bar(labels, vals, color="#2a78d6", width=0.55)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}s", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=10, color="#0b0b0b")
    ax.set_ylabel("one RL iteration, k=8 (seconds)", color="#52514e")
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color="#e7e7e4", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.suptitle(f"SS13 — waterfall | predicted: {predicted}",
                 fontsize=9.5, x=0.01, ha="left")
    ax.set_title(f"measured: {measured}", fontsize=8.5, loc="left",
                 color="#52514e", pad=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(args.out, "ss13_waterfall.png"), dpi=300,
                facecolor="#fcfcfb")
    json.dump({"rungs": rungs, "speedup": speedup},
              open(os.path.join(args.out, "result.json"), "w"), indent=2)
    write_manifest(args.out, "SS13", "iteration waterfall, rung by rung",
                   predicted=predicted, measured=measured,
                   mechanism=("nearly all the win is batched rollouts filling "
                              "env gaps + amortizing weight reads; engine swap "
                              "alone and data-path tweaks measured ≈ nothing "
                              "(SS2/SS9/SS10); ckpt-off trims the update"),
                   config={"k": args.k, "merged": True},
                   rerun_cmd="scripts/ss13_session.sh (staged)")
    print(f"[SS13] {measured}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["merge", "rollout_hf", "rollout_vllm", "update",
                             "assemble"])
    ap.add_argument("--sft-adapter",
                    default="/scratch/madan.pa/pennypilot/rloo50_v2/policy")
    ap.add_argument("--merged-dir",
                    default="/scratch/madan.pa/pennypilot/rloo50_merged")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--predicted", default="")
    ap.add_argument("--out", default="benchmarks/artifacts/ss13")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"merge": stage_merge, "rollout_hf": stage_rollout_hf,
     "rollout_vllm": stage_rollout_vllm, "update": stage_update,
     "assemble": stage_assemble}[args.stage](args)


if __name__ == "__main__":
    main()
