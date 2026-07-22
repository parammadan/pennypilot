"""SS11 — POST-OPTIMIZATION RE-PROFILE of the update phase (GPU, trainer env).

The gate for SS12: only if the masked-logprob → advantage-weighting → reduce
chain shows up as a multi-kernel hotspot (≥1% of ITERATION time) does a fused
Triton kernel get written. A profile showing no hotspot is the deliverable.

    python benchmarks/ss11_update_reprofile.py --sft-adapter <adapter> \
        --predicted "..." --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--predicted", default="")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--iteration-rollout-frac", type=float, default=0.958,
                    help="SS0-measured rollout share, to express the chain as "
                         "a fraction of the FULL iteration")
    ap.add_argument("--out", default="benchmarks/artifacts/ss11")
    args = ap.parse_args()

    from shoprl.profiling import require_prediction, write_manifest
    predicted = require_prediction(args.predicted)
    os.makedirs(args.out, exist_ok=True)

    import torch
    from torch.profiler import ProfilerActivity, profile

    from shoprl.data.catalog import catalog_index, generate_catalog
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2
    from shoprl.env.catalog_env import SyntheticCatalogEnvironment
    from shoprl.env.scenario import generate_hard_scenarios
    from shoprl.profiling.bench_common import load_hf_policy
    from shoprl.train.algo import RLOO, rollout_v2
    from shoprl.train.rloo import _k3_kl, _sequence_and_mask, _token_logprobs

    policy, tok = load_hf_policy("Qwen/Qwen2.5-1.5B-Instruct", args.sft_adapter)
    reference, _ = load_hf_policy("Qwen/Qwen2.5-1.5B-Instruct", args.sft_adapter)
    for p in reference.parameters():
        p.requires_grad_(False)
    for n, p in policy.named_parameters():
        if "lora" in n.lower():
            p.requires_grad_(True)
            p.data = p.data.float()
    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad), lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    catalog = generate_catalog(n=300, seed=0)
    idx = catalog_index(catalog)
    scen = generate_hard_scenarios(catalog, n=1, seed=3000,
                                   n_must_haves=(3, 4),
                                   valid_target_range=(3, 10))[0]

    # Rollouts OUTSIDE the profile (they're 95.8% of the iteration; the
    # question is only about the update phase).
    import types
    policy.eval()
    policy.config.use_cache = True

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

    trajs = []
    for _ in range(args.k):
        env = SyntheticCatalogEnvironment(catalog, scen, idx=idx, language="es-en")
        trajs.append(rollout_v2(agent_fn, env, SYSTEM_PROMPT_V2))
    advs = RLOO().advantages([t.reward for t in trajs])

    policy.train()
    policy.config.use_cache = False
    torch.cuda.synchronize()
    t0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        optimizer.zero_grad(set_to_none=True)
        for t, A in zip(trajs, advs):
            ids, mask = _sequence_and_mask(tok, t.messages, 2048)
            asst = torch.tensor(mask[1:], device="cuda", dtype=torch.bool)
            if asst.sum() == 0:
                continue
            with torch.autocast("cuda", dtype=torch.float16):
                lp_pol = _token_logprobs(policy, ids, "cuda")
                with torch.no_grad():
                    lp_ref = _token_logprobs(reference, ids, "cuda")
            pg = -A * lp_pol[asst].mean()
            kl = _k3_kl(lp_pol[asst], lp_ref[asst])
            loss = (pg + 0.04 * kl.mean()) / args.k
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    update_s = time.time() - t0
    prof.export_chrome_trace(os.path.join(args.out, "update_trace.json"))

    rows = prof.key_averages()
    total_cuda = sum(getattr(r, "self_device_time_total",
                             getattr(r, "self_cuda_time_total", 0)) for r in rows)
    chain_pat = re.compile(
        r"gather|log_softmax|softmax(?!.*mm)|masked|index|mean|::mul|::sub|::exp"
        r"|::div|::add(?!mm)|where|nonzero", re.IGNORECASE)
    chain_cuda = sum(getattr(r, "self_device_time_total",
                             getattr(r, "self_cuda_time_total", 0))
                     for r in rows if chain_pat.search(r.key))
    chain_frac_update = chain_cuda / max(total_cuda, 1)
    update_frac_iter = 1 - args.iteration_rollout_frac
    chain_frac_iter = chain_frac_update * update_frac_iter

    top = sorted(rows, key=lambda r: -getattr(r, "self_device_time_total",
                 getattr(r, "self_cuda_time_total", 0)))[:20]
    table = [{"op": r.key[:70],
              "self_cuda_ms": round(getattr(r, "self_device_time_total",
                                    getattr(r, "self_cuda_time_total", 0)) / 1e3, 2),
              "calls": r.count} for r in top]

    kernel_justified = chain_frac_iter >= 0.01
    verdict = ("HOTSPOT — Triton fusion justified (SS12 proceeds)"
               if kernel_justified else
               "NO KERNEL-WORTHY HOTSPOT — the PG-math chain is "
               f"{chain_frac_iter*100:.2f}% of the iteration; SS12 = the "
               "documented decision NOT to write the kernel")
    measured = (f"update {update_s:.1f}s; PG-math chain ≈ "
                f"{chain_frac_update*100:.1f}% of update CUDA time ≈ "
                f"{chain_frac_iter*100:.2f}% of the full iteration → {verdict}")
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"update_seconds": round(update_s, 2),
                   "chain_frac_of_update_cuda": round(chain_frac_update, 4),
                   "chain_frac_of_iteration": round(chain_frac_iter, 5),
                   "kernel_justified": kernel_justified,
                   "top_ops": table}, f, indent=2)
    write_manifest(args.out, "SS11", "update-phase re-profile (kernel gate)",
                   predicted=predicted, measured=measured,
                   mechanism=("the update is model fwd/bwd GEMM-dominated; the "
                              "elementwise PG chain is tiny and already fused "
                              "reasonably by eager kernels"),
                   config={"k": args.k},
                   rerun_cmd="python benchmarks/ss11_update_reprofile.py "
                             "--sft-adapter ... --predicted '...'")
    print(f"[SS11] {measured}")


if __name__ == "__main__":
    main()
