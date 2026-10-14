"""Tool-decision + mechanism probes for one checkpoint (paper eval A).

Three measurements, all greedy, all answers saved for audit:

  1. Tool decision under the deployment prompt. ON-task = the opening user
     message of each held-out paper scenario (es-en; a store action is the
     right response). OFF-task = GENERAL_EVAL (40, en+es) + 200 held-out Dolly
     prompts (disjoint from rehearsal; prose is the right response).
     Reports tool-call rate on each side and decision accuracy.
  2. Prompt probes on the OFF-task set: is the collapse tied to the
     deployment prompt? Conditions: deploy | none (no system message) |
     plain (generic assistant) | deploy+direct (deploy + explicit instruction
     to answer non-shopping questions directly).
  3. Answer correctness (does the model still KNOW and SAY the answer?):
     TriviaQA (300, en; any gold alias in the normalized reply) and MGSM
     (6 langs x 50; last number == gold), under deploy and none prompts.
     A store-action reply counts as wrong; correct_given_prose separates
     knowledge from format.
  4. LoRA-scale sweep: adapter scaled by alpha in {0,.25,.5,.75,1}, deploy
     prompt, GENERAL_EVAL only. alpha=0 is the base model, alpha=1 the
     checkpoint: where does the collapse set in?

    python scripts/paper/eval_behavior.py --adapter <dir|base> --out <json>
"""
import argparse
import json
import os

import torch

from shoprl.actions import parse_agent_action
from shoprl.data.general_chat import GENERAL_EVAL
from shoprl.data.prompts_v2 import SYSTEM_PROMPT_CHAT_MIN
from shoprl.env.catalog_env import SyntheticCatalogEnvironment
from shoprl.eval.qa_score import correct
from shoprl.eval.splits import paper_eval_split
from shoprl.profiling.bench_common import load_hf_policy
from shoprl.repro import seed_everything, write_manifest

PLAIN = "You are a helpful assistant."
DIRECT = ("\n\nIf the user's message is not about finding, comparing or buying "
          "a product, answer it directly in prose and do not take any store action.")
PROMPTS = {"deploy": SYSTEM_PROMPT_CHAT_MIN, "none": None, "plain": PLAIN,
           "deploy+direct": SYSTEM_PROMPT_CHAT_MIN + DIRECT}
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def is_tool(text: str) -> bool:
    return parse_agent_action(text).ok


def set_lora_scale(model, alpha: float, base: dict) -> None:
    for name, mod in model.named_modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
            for k in mod.scaling:
                mod.scaling[k] = base[(name, k)] * alpha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="base")
    ap.add_argument("--offtask", default="src/shoprl/data/offtask_dolly200.jsonl")
    ap.add_argument("--n-ontask", type=int, default=128)
    ap.add_argument("--qa", default="src/shoprl/data/qa_correctness.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--qa-max-new-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_manifest(os.path.dirname(args.out) or ".", args, kind="eval_behavior")
    seed_everything(0)
    adapter = None if args.adapter == "base" else args.adapter
    model, tok = load_hf_policy(args.model, adapter)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    @torch.no_grad()
    def gen(sys_prompt, qs, bs=16, max_new=None):
        outs = []
        for i in range(0, len(qs), bs):
            batch = []
            for q in qs[i:i + bs]:
                m = ([{"role": "system", "content": sys_prompt}] if sys_prompt else [])
                batch.append(tok.apply_chat_template(
                    m + [{"role": "user", "content": q}],
                    add_generation_prompt=True, tokenize=False))
            enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
            out = model.generate(**enc, max_new_tokens=max_new or args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
            outs += [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                     for o in out]
        return outs

    catalog, scenarios = paper_eval_split(n=args.n_ontask)
    ontask = []
    for sc in scenarios:
        env = SyntheticCatalogEnvironment(catalog, sc, language="es-en")
        ontask.append(env.reset())
    gen40 = [x["q"] for x in GENERAL_EVAL]
    dolly = [json.loads(l)["q"] for l in open(args.offtask)]
    offtask = gen40 + dolly

    rec = {"model": args.model, "adapter": args.adapter, "answers": {}}

    a_on = gen(PROMPTS["deploy"], ontask)
    on_rate = sum(map(is_tool, a_on)) / len(a_on)
    rec["answers"]["ontask/deploy"] = [{"q": q, "a": a} for q, a in zip(ontask, a_on)]

    probes = {}
    for name, sp in PROMPTS.items():
        a = gen(sp, offtask)
        rec["answers"][f"offtask/{name}"] = [{"q": q, "a": x} for q, x in zip(offtask, a)]
        t = [is_tool(x) for x in a]
        probes[name] = {"tool_rate_all": sum(t) / len(t),
                        "tool_rate_gen40": sum(t[:len(gen40)]) / len(gen40),
                        "tool_rate_dolly": sum(t[len(gen40):]) / len(dolly)}
    off_rate = probes["deploy"]["tool_rate_all"]
    n_on, n_off = len(ontask), len(offtask)
    rec["tool_decision"] = {
        "ontask_tool_rate": on_rate, "offtask_tool_rate": off_rate,
        "n_ontask": n_on, "n_offtask": n_off,
        "accuracy": (on_rate * n_on + (1 - off_rate) * n_off) / (n_on + n_off)}
    rec["prompt_probes"] = probes

    qa = [json.loads(l) for l in open(args.qa)]
    corr = {}
    for name in ("deploy", "none"):
        a = gen(PROMPTS[name], [r["q"] for r in qa], bs=8, max_new=args.qa_max_new_tokens)
        rec["answers"][f"qa/{name}"] = [{"q": r["q"], "gold": r["gold"], "a": x}
                                        for r, x in zip(qa, a)]
        by = {}
        for r, x in zip(qa, a):
            k = r["set"] if r["set"] == "triviaqa" else f"mgsm_{r['lang']}"
            d = by.setdefault(k, {"n": 0, "correct": 0, "tool": 0, "prose_correct": 0})
            t, c = is_tool(x), correct(r, x) and not is_tool(x)
            d["n"] += 1; d["correct"] += c; d["tool"] += t; d["prose_correct"] += c
        for k, d in by.items():
            prose = d["n"] - d["tool"]
            d["accuracy"] = d["correct"] / d["n"]
            d["tool_rate"] = d["tool"] / d["n"]
            d["correct_given_prose"] = d["prose_correct"] / prose if prose else None
        mg = [v for k, v in by.items() if k.startswith("mgsm_")]
        by["mgsm_all"] = {"accuracy": sum(v["correct"] for v in mg) / sum(v["n"] for v in mg),
                          "tool_rate": sum(v["tool"] for v in mg) / sum(v["n"] for v in mg)}
        corr[name] = by
    rec["correctness"] = corr

    sweep = {}
    if adapter:
        base = {(n, k): v for n, m in model.named_modules()
                if hasattr(m, "scaling") and isinstance(m.scaling, dict)
                for k, v in m.scaling.items()}
        for al in ALPHAS:
            set_lora_scale(model, al, base)
            a = gen(PROMPTS["deploy"], gen40)
            rec["answers"][f"sweep/{al}"] = [{"q": q, "a": x} for q, x in zip(gen40, a)]
            sweep[str(al)] = sum(map(is_tool, a)) / len(a)
        set_lora_scale(model, 1.0, base)
    rec["lora_sweep_tool_rate_gen40"] = sweep

    json.dump(rec, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"[behavior] {args.adapter}: decision={rec['tool_decision']} "
          f"probes={ {k: round(v['tool_rate_all'], 3) for k, v in probes.items()} } "
          f"triviaqa(deploy/none)={corr['deploy']['triviaqa']['accuracy']:.3f}/"
          f"{corr['none']['triviaqa']['accuracy']:.3f} "
          f"mgsm(deploy/none)={corr['deploy']['mgsm_all']['accuracy']:.3f}/"
          f"{corr['none']['mgsm_all']['accuracy']:.3f} sweep={sweep}")


if __name__ == "__main__":
    main()
