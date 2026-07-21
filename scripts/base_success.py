"""SCENARIO HARDNESS GATE (GPU) — base-model success on hard scenarios.

Measures the UN-fine-tuned instruct model's task success on the v2 hard
scenarios, using the same system prompt SFT/RL will use. Acceptance band:
10-40%. Below 10%: too hard (SFT can't bootstrap) — above 40%: too easy
(v1 saturation repeats). Either way outside the band -> redesign scenarios,
do NOT proceed to SFT.

    python scripts/base_success.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --n 64 --out /scratch/madan.pa/pennypilot/gate/base_success.json

Success = a legitimately carted item that satisfies every hidden constraint
(value_quality > 0, no permission violation). CPU-free of charge: everything
except .generate() is the tested environment plane.
"""
from __future__ import annotations

import argparse
import json
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=11, help="hard-scenario seed")
    ap.add_argument("--language", default="es-en", choices=["en", "es", "es-en"])
    ap.add_argument("--catalog-size", type=int, default=300)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default="runs/gate/base_success.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from shoprl.data.catalog import catalog_index, generate_catalog
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2
    from shoprl.env.catalog_env import SyntheticCatalogEnvironment
    from shoprl.env.scenario import generate_hard_scenarios

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="sdpa").to("cuda").eval()

    catalog = generate_catalog(n=args.catalog_size, seed=0)
    idx = catalog_index(catalog)
    scenarios = generate_hard_scenarios(catalog, n=args.n, seed=args.seed)

    @torch.no_grad()
    def agent_turn(messages: list[dict]) -> str:
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      tokenize=True, return_dict=True,
                                      return_tensors="pt")
        ids = enc["input_ids"].to("cuda")
        out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    results = []
    t0 = time.time()
    for i, scen in enumerate(scenarios):
        env = SyntheticCatalogEnvironment(catalog, scen, idx=idx,
                                          max_turns=args.max_turns,
                                          language=args.language)
        opener = env.reset()
        messages = [{"role": "system", "content": SYSTEM_PROMPT_V2},
                    {"role": "user", "content": opener}]
        done = False
        while not done:
            action = agent_turn(messages)
            step = env.execute_text(action)
            done = step.done
            messages += [{"role": "assistant", "content": action},
                         {"role": "user", "content": step.observation}]
        out = env.calculate_outcome()
        success = out.value_quality > 0 and out.acted_without_permission == 0
        results.append({
            "scenario_id": scen.scenario_id, "success": bool(success),
            "value_quality": out.value_quality,
            "violation": bool(out.acted_without_permission),
            "asked": any("revealed" in t.note for t in env.turns),
            "invalid_actions": sum(not t.valid for t in env.turns),
            "turns": len(env.turns),
        })
        print(f"[{i+1:>3}/{args.n}] {scen.scenario_id} "
              f"success={success} value={out.value_quality:.2f} "
              f"viol={out.acted_without_permission:.0f} turns={len(env.turns)}")

    n = len(results)
    rate = sum(r["success"] for r in results) / n
    verdict = ("PASS (10-40% band) — proceed to SFT" if 0.10 <= rate <= 0.40 else
               "FAIL LOW (<10%) — scenarios too hard, redesign" if rate < 0.10 else
               "FAIL HIGH (>40%) — scenarios too easy, redesign")
    summary = {
        "model": args.model, "dtype": args.dtype, "language": args.language,
        "n": n, "success_rate": round(rate, 4),
        "violation_rate": round(sum(r["violation"] for r in results) / n, 4),
        "ask_rate": round(sum(r["asked"] for r in results) / n, 4),
        "mean_invalid_actions": round(
            sum(r["invalid_actions"] for r in results) / n, 2),
        "wall_seconds": round(time.time() - t0, 1),
        "gate_verdict": verdict,
        "episodes": results,
    }
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== HARDNESS GATE: success {rate:.1%} on n={n} -> {verdict}")
    print(f"    (violations {summary['violation_rate']:.1%}, "
          f"asks {summary['ask_rate']:.1%}, "
          f"{summary['wall_seconds']}s) -> {args.out}")


if __name__ == "__main__":
    main()
