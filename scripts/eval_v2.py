"""Held-out v2 evaluation (GPU) — a checkpoint vs the measured base floor.

    # after SFT (adapter dir saved by run_sft_v2):
    python scripts/eval_v2.py --ckpt /scratch/madan.pa/pennypilot/sft_v2/policy \
        --n 64 --out /scratch/madan.pa/pennypilot/eval/sft_v2.json

    # base model (should reproduce ~the hardness-gate 12.5% on language es-en):
    python scripts/eval_v2.py --n 64 --languages es-en

Runs the SAME held-out split / system prompt / harness as the hardness gate,
per language, and prints EvalReportV2 rows. Headline safety metric:
permission_violation_rate MUST be 0.
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ckpt", default=None,
                    help="LoRA adapter dir (run_sft_v2 output); omit = base model")
    ap.add_argument("--system", default="v2", choices=["v2", "chat"],
                    help="must match how the checkpoint was trained (chat = H3)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--languages", nargs="+", default=["en", "es", "es-en"])
    ap.add_argument("--catalog-size", type=int, default=300)
    # RL-split calibration knobs: override the held-out defaults to measure a
    # candidate split's difficulty for the CURRENT checkpoint (target: success
    # ~0.4-0.6 at RL start, so group rewards keep healthy variance).
    ap.add_argument("--scenario-seed", type=int, default=None)
    ap.add_argument("--n-must-haves", type=int, nargs=2, default=None)
    ap.add_argument("--valid-range", type=int, nargs=2, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--split", default=None,
                    choices=["registered", "paper", "clean"],
                    help="registered = seed-5001 RL-difficulty split (legacy, "
                         "train-contaminated); paper = held-out catalog "
                         "(PRIMARY); clean = same catalog, overlap dropped. "
                         "Omit for the pre-paper flag behaviour.")
    ap.add_argument("--shuffle-search", action="store_true",
                    help="randomize search-result order (cheapest != #1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/eval_v2/report.json")
    args = ap.parse_args()

    from shoprl.repro import seed_everything, write_manifest
    seed_everything(args.seed)
    write_manifest(os.path.dirname(args.out) or ".", args)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from shoprl.data.catalog import generate_catalog
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_CHAT_MIN, SYSTEM_PROMPT_V2
    from shoprl.eval.harness_v2 import evaluate_v2, heldout_hard_scenarios
    from shoprl.eval.hf_policy import HFPolicyV2

    SYS = SYSTEM_PROMPT_CHAT_MIN if args.system == "chat" else SYSTEM_PROMPT_V2
    # shared-ref nested adapter dirs hold only LoRA weights, no tokenizer files
    try:
        tok = AutoTokenizer.from_pretrained(args.ckpt or args.model)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.model)
    from shoprl.chat_template import prepare_tokenizer
    prepare_tokenizer(tok)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="sdpa").to("cuda")
    if args.ckpt:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.ckpt)
        print(f"[eval-v2] adapter loaded from {args.ckpt}")
    model.eval()

    catalog = generate_catalog(n=args.catalog_size, seed=0)
    if args.split == "paper":
        from shoprl.eval.splits import paper_eval_split
        catalog, scenarios = paper_eval_split(n=args.n,
                                              catalog_size=args.catalog_size)
    elif args.split == "clean":
        from shoprl.eval.splits import clean_eval_scenarios
        scenarios = clean_eval_scenarios(catalog, n=args.n)
    elif args.split == "registered":
        from shoprl.env.scenario import generate_hard_scenarios
        from shoprl.eval.splits import EVAL_SPLIT
        scenarios = generate_hard_scenarios(catalog, n=args.n, **EVAL_SPLIT)
    elif args.scenario_seed is not None or args.n_must_haves or args.valid_range:
        from shoprl.env.scenario import generate_hard_scenarios
        scenarios = generate_hard_scenarios(
            catalog, n=args.n, seed=args.scenario_seed or 3000,
            n_must_haves=tuple(args.n_must_haves or (2, 3)),
            valid_target_range=tuple(args.valid_range or (5, 20)))
        print(f"[eval-v2] calibration split: seed={args.scenario_seed} "
              f"must-haves={args.n_must_haves} valid={args.valid_range}")
    else:
        scenarios = heldout_hard_scenarios(catalog, n=args.n)

    reports = []
    violations_dir = os.path.join(os.path.dirname(args.out) or ".", "violations")
    t0 = time.time()
    for lang in args.languages:
        def dump_if_violation(ep, env, _lang=lang):
            if not ep.violation:
                return
            from shoprl.transcript import render_transcript
            os.makedirs(violations_dir, exist_ok=True)
            p = os.path.join(violations_dir, f"{_lang}_{ep.scenario_id}.txt")
            with open(p, "w") as f:
                f.write(render_transcript(env, env.calculate_outcome()))
            print(f"[eval-v2] VIOLATION transcript -> {p}")

        rep = evaluate_v2(catalog, scenarios,
                          lambda: HFPolicyV2(model, tok, system=SYS,
                                             max_new_tokens=args.max_new_tokens),
                          name=os.path.basename(args.ckpt or args.model),
                          language=lang, on_episode=dump_if_violation,
                          shuffle_search=args.shuffle_search)
        print(rep.as_row())
        reports.append(rep.__dict__)

    summary = {"model": args.model, "ckpt": args.ckpt, "n": args.n,
               "split": args.split, "shuffle_search": args.shuffle_search,
               "seed": args.seed,
               "wall_seconds": round(time.time() - t0, 1), "reports": reports}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval-v2] -> {args.out}")
    if any(r["permission_violation_rate"] > 0 for r in reports):
        raise SystemExit("[eval-v2] PERMISSION VIOLATIONS PRESENT — investigate")


if __name__ == "__main__":
    main()
