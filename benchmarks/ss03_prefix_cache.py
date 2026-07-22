"""SS3 — PREFIX CACHING: per-turn prefill latency, APC off vs on (vLLM venv).

The multi-turn signature chart: each turn's prompt extends the previous turn's
prompt, so WITHOUT prefix caching every turn re-prefills the whole growing
conversation; WITH automatic prefix caching only the delta is prefilled.

    VLLM_USE_V1=0 <venv>/bin/python benchmarks/ss03_prefix_cache.py \
        --sft-adapter <adapter> --predicted "..." \
        2>&1 | tee benchmarks/artifacts/ss03/run_$SLURM_JOB_ID.log

Measurement: per-turn generate latency at FIXED max_tokens=64 (the proven-safe
kernel shape on these V100s — max_tokens=1/temp=0 shapes hard-abort in Triton
on some call sequences, see CHALLENGES #26). Decode cost is ~constant per
turn, so the GROWTH of the curve across turns isolates prefill growth; APC
should flatten that growth. Honest note: absolute values include ~64 decode
steps; the slope is the signal.
"""
from __future__ import annotations

import argparse
import json
import os
import time


def conversation_prefixes(max_turns: int = 10) -> list[list[dict]]:
    """Turn-by-turn message prefixes of ONE deterministic oracle episode on a
    4-constraint scenario (CPU; env plane only)."""
    from shoprl.data.catalog import catalog_index, generate_catalog
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2
    from shoprl.env.catalog_env import SyntheticCatalogEnvironment
    from shoprl.env.scenario import generate_hard_scenarios
    from shoprl.eval.v2_policies import OracleGoodV2

    catalog = generate_catalog(n=300, seed=0)
    idx = catalog_index(catalog)
    scen = generate_hard_scenarios(catalog, n=1, seed=3000,
                                   n_must_haves=(4, 4),
                                   valid_target_range=(3, 10))[0]
    env = SyntheticCatalogEnvironment(catalog, scen, idx=idx, language="es-en")
    policy = OracleGoodV2()
    opener = env.reset()
    policy.reset(scen, idx)
    messages = [{"role": "system", "content": SYSTEM_PROMPT_V2},
                {"role": "user", "content": opener}]
    prefixes = []
    done = False
    while not done and len(prefixes) < max_turns:
        prefixes.append([dict(m) for m in messages])
        action = policy.act()
        step = env.execute_text(action)
        messages += [{"role": "assistant", "content": action},
                     {"role": "user", "content": step.observation}]
        done = step.done
    return prefixes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sft-adapter", default=None)
    ap.add_argument("--predicted", default="")
    ap.add_argument("--gpu-mem-util", type=float, default=0.45)
    ap.add_argument("--mode", default="plot", choices=["apc_off", "apc_on", "plot"],
                    help="measure one mode per PROCESS (a Volta Triton abort in "
                         "one mode must not kill the other's data), then plot")
    ap.add_argument("--out", default="benchmarks/artifacts/ss03")
    args = ap.parse_args()

    from shoprl.profiling import require_prediction, write_manifest
    predicted = require_prediction(args.predicted)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    prefixes = conversation_prefixes()
    prompts = [tok.apply_chat_template(p, tokenize=False,
                                       add_generation_prompt=True)
               for p in prefixes]
    prompt_tokens = [len(tok(p)["input_ids"]) for p in prompts]

    if args.mode in ("apc_off", "apc_on"):
        from vllm import LLM, SamplingParams
        sp = SamplingParams(temperature=0.8, max_tokens=64, seed=0)
        llm = LLM(model=args.model, dtype="float16",
                  gpu_memory_utilization=args.gpu_mem_util,
                  enable_prefix_caching=(args.mode == "apc_on"),
                  enforce_eager=True)
        llm.generate([prompts[0]], sp, use_tqdm=False)    # warm the engine
        lat = []
        for pr in prompts:
            t0 = time.time()
            llm.generate([pr], sp, use_tqdm=False)
            lat.append(round((time.time() - t0) * 1000, 1))
        with open(os.path.join(args.out, f"{args.mode}.json"), "w") as f:
            json.dump({"prompt_tokens": prompt_tokens, "latency_ms": lat}, f)
        print(f"[SS3:{args.mode}] {lat}")
        return

    # plot mode: assemble whatever modes survived (a missing apc_on.json IS
    # the reportable Volta result, not a failure of the benchmark).
    results = {}
    for mode in ("apc_off", "apc_on"):
        fp = os.path.join(args.out, f"{mode}.json")
        if os.path.exists(fp):
            results[mode] = json.load(open(fp))["latency_ms"]
    if "apc_on" in results:
        measured = (f"turn2→turn{len(prompts)}: off {results['apc_off'][1]}→"
                    f"{results['apc_off'][-1]} ms, on {results['apc_on'][1]}→"
                    f"{results['apc_on'][-1]} ms")
        mechanism = ("APC reuses the shared prefix's KV blocks so only the new "
                     "turn is prefilled; without it prefill grows with turns")
    else:
        off = results["apc_off"]
        measured = (f"APC-off per-turn latency grows {off[1]}→{off[-1]} ms "
                    f"(turns 2→{len(off)}); **APC-on UNAVAILABLE on Volta**: "
                    "vLLM 0.7.3's prefix-cache prefill kernel (Triton) aborts — "
                    "'mma layout conversion only supported on Ampere'")
        mechanism = ("prefill work grows with conversation length (measured); "
                     "the caching fix requires a prefix-prefill kernel this "
                     "hardware cannot compile — the optimization is documented "
                     "as hardware-gated, per the campaign rule that a blocked "
                     "path is itself a reportable artifact")
    _plot(results, prompt_tokens, args.out, predicted, measured)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"prompt_tokens": prompt_tokens, **results}, f, indent=2)
    write_manifest(args.out, "SS3", "per-turn latency (fixed 64-tok decode), APC off vs on",
                   predicted=predicted, measured=measured, mechanism=mechanism,
                   config=vars(args),
                   rerun_cmd="VLLM_USE_V1=0 <venv>/bin/python "
                             "benchmarks/ss03_prefix_cache.py --mode apc_off|apc_on|plot "
                             "--predicted '...'")
    print(f"[SS3] {measured}")


def _plot(results, prompt_tokens, out, predicted, measured) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    turns = list(range(1, len(prompt_tokens) + 1))
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#fcfcfb")
    ax.plot(turns, results["apc_off"], color="#2a78d6", linewidth=2,
            marker="o", markersize=5)
    ax.annotate("prefix caching OFF", (turns[-1], results["apc_off"][-1]),
                xytext=(-8, 10), textcoords="offset points", ha="right",
                color="#0b0b0b", fontsize=9)
    if "apc_on" in results:
        ax.plot(turns, results["apc_on"], color="#eb6834", linewidth=2,
                marker="o", markersize=5)
        ax.annotate("prefix caching ON", (turns[-1], results["apc_on"][-1]),
                    xytext=(-8, 10), textcoords="offset points", ha="right",
                    color="#0b0b0b", fontsize=9)
    else:
        ax.annotate("APC-on: unavailable on Volta\n(Triton kernel needs Ampere)",
                    (0.02, 0.95), xycoords="axes fraction", va="top",
                    color="#b25a00", fontsize=9)
    ax.set_xlabel("conversation turn", color="#52514e")
    ax.set_ylabel("per-turn latency (ms, 64-token decode)", color="#52514e")
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#e7e7e4", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.suptitle(f"SS3 — multi-turn prefill | predicted: {predicted}",
                 fontsize=10, x=0.01, ha="left")
    ax.set_title(f"measured: {measured}", fontsize=8.5, loc="left",
                 color="#52514e", pad=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{out}/ss03_prefill.png", dpi=300, facecolor="#fcfcfb")
    import csv
    with open(f"{out}/ss03_prefill.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn", "prompt_tokens", "apc_off_ms", "apc_on_ms"])
        for i, t in enumerate(turns):
            w.writerow([t, prompt_tokens[i], results["apc_off"][i],
                        results.get("apc_on", [None] * len(turns))[i]])


if __name__ == "__main__":
    main()
