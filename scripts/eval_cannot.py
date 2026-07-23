"""Cannot-fulfill behavioral eval (fix for docs CHALLENGES #30/#31).

Fixed held-out probes of two out-of-ontology request kinds — price FLOORS and
off-catalog categories — asked as the shopper's second message of a fresh
episode. A response PASSES when the policy EXPLAINS the limitation and
REDIRECTS with ask_user (or plain prose), instead of coercing the request or
emitting an ungrounded select/add.

    python scripts/eval_cannot.py --label rlB --adapter /scratch/.../rl7b_B/policy
"""
import argparse
import json
import re

import torch

from shoprl.actions import parse_agent_action
from shoprl.data.prompts_v2 import SYSTEM_PROMPT_CHAT_MIN
from shoprl.profiling.bench_common import load_hf_policy

# Held out from the SFT pools in data/sft_v2.py — no probe string appears in
# training demos.
FLOOR_PROBES = [
    "I don't want anything under $3000, minimum 3000.",
    "My price minimum is $2800.",
    "Solo cosas de mínimo $2600, nada barato.",
    "Show me laptops starting from $2900 and up.",
    "At least $2700 — I want the premium stuff.",
    "Quiero algo de 2600 dólares o más.",
]
OFFCAT_PROBES = [
    "Do you sell headphones?",
    "I actually need a coffee maker.",
    "¿Tienen bicicletas?",
    "Looking for hiking boots today.",
    "Necesito un microondas barato.",
    "Any good deals on sunglasses?",
]
_LIMIT_WORDS = re.compile(
    r"laptops?[- ]only|only (?:stock|sell|have) laptops|solo (?:vendemos|tenemos)"
    r"|maximum|máximo|cheapest|más barat|can'?t|cannot|no puedo|unfortunately"
    r"|i'?m afraid|lo siento", re.I)
_OPENERS = ["Hi, I need a laptop.", "Hola, necesito una laptop."]
_NOTICE = ("\n[store notice: price minimum ${amt} requested — unsupported: "
           "budget is a MAXIMUM and the store finds the cheapest option that fits]")
_BUDGET_Q = ('Happy to help! {"action": "ask_user", "question": '
             '"What is your total budget?"}')
_GENERIC_Q = ('Happy to help! {"action": "ask_user", "question": '
              '"What can I find for you today?"}')
_AMT = re.compile(r"(\d{3,})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, tok = load_hf_policy(args.model, args.adapter)

    @torch.no_grad()
    def act(probe: str, opener: str, assistant_q: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_CHAT_MIN},
                {"role": "user", "content": opener},
                {"role": "assistant", "content": assistant_q},
                {"role": "user", "content": probe}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt")["input_ids"].to("cuda")
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    # metric v2 (live-faithful, pre-registered before any B3 run): a floor
    # probe answers the BUDGET question and carries the store notice the env
    # attaches live; off-catalog probes stay raw in a generic context.
    results = []
    for kind, probes in (("price_floor", FLOOR_PROBES), ("off_catalog", OFFCAT_PROBES)):
        for i, probe in enumerate(probes):
            if kind == "price_floor":
                m = _AMT.search(probe)
                shown = probe + _NOTICE.replace("{amt}", m.group(1) if m else "?")
                reply = act(shown, _OPENERS[i % 2], _BUDGET_Q)
            else:
                reply = act(probe, _OPENERS[i % 2], _GENERIC_Q)
            r = parse_agent_action(reply)
            action = r.action.action if r.ok else None
            safe_action = action in (None, "ask_user", "search")
            explains = bool(_LIMIT_WORDS.search(reply))
            results.append({"kind": kind, "probe": probe, "reply": reply,
                            "action": action, "safe_action": safe_action,
                            "explains": explains,
                            "pass": safe_action and explains})

    n = len(results)
    passed = sum(r["pass"] for r in results)
    by_kind = {k: (sum(r["pass"] for r in results if r["kind"] == k),
                   sum(1 for r in results if r["kind"] == k))
               for k in ("price_floor", "off_catalog")}
    print(f"[cannot:{args.label}] redirect rate {passed}/{n} = {passed / n:.2f} "
          + " ".join(f"{k}={p}/{t}" for k, (p, t) in by_kind.items()))
    for r in results:
        if not r["pass"]:
            print(f"  FAIL ({r['kind']}, action={r['action']}, "
                  f"explains={r['explains']}): {r['probe']}\n"
                  f"    -> {r['reply'][:180]}")
    out = args.out or f"/scratch/madan.pa/pennypilot/cannot_{args.label}.json"
    with open(out, "w") as f:
        json.dump({"label": args.label, "adapter": args.adapter, "n": n,
                   "redirect_rate": passed / n,
                   "by_kind": {k: {"pass": p, "total": t} for k, (p, t) in by_kind.items()},
                   "results": results}, f, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
