"""HUMAN-IN-THE-LOOP demo — YOU are the shopper, the trained model serves you.

    python scripts/demo_human.py --policy-url http://localhost:8765

You get a secret shopping brief (budget + must-haves). The agent asks you
questions in the browser+terminal; you answer IN THE TERMINAL in any language
(press Enter for a suggested reply). You decide permission yourself — the
model cannot cart anything without your explicit yes.

Honesty notes: the store's ground truth (which items are valid/cheapest)
follows your BRIEF, so stay roughly on-brief when revealing numbers; your
words are what the model sees either way. Your y/n at the permission prompt
is the real authority (the programmatic judge is bypassed in human mode).
"""
from __future__ import annotations

import argparse
import json


class HumanConversation:
    """The ConversationModel seam, played by a person at the terminal."""

    def __init__(self, scenario, phrase, phrase_es):
        self.scenario = scenario
        self._phrase = phrase

    def _ask(self, prompt: str, default: str) -> str:
        try:
            text = input(f"\n🧑 {prompt}\n   [Enter = “{default}”] > ").strip()
        except EOFError:
            text = ""
        return text or default

    def utter(self, intent: str, scenario, accepted=None) -> str:
        if intent == "greet":
            return self._ask("Your opening request to the assistant "
                             "(any language, DON'T reveal budget/specs yet):",
                             "Necesito una laptop, can you help?")
        if intent == "budget":
            return self._ask(f"It asked your budget — your brief says "
                             f"${scenario.hidden_budget:.0f}. Your words:",
                             f"My budget is about ${scenario.hidden_budget:.0f}.")
        if intent == "permission":   # words only; the DECISION was yours already
            return ("Yes, please add it." if accepted
                    else "No, that one doesn't work for me.")
        return self._ask("It said/asked something generic — your reply:",
                         "Could you help me find the right one?")

    def utter_constraint(self, key: str, value) -> str:
        return self._ask(f"It asked about requirements — your brief includes "
                         f"{key} = {value}. Your words:", self._phrase(key, value))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-url", default="http://localhost:8765")
    ap.add_argument("--scenario-seed", type=int, default=11)
    ap.add_argument("--scenario-index", type=int, default=4)
    ap.add_argument("--headed", action="store_true", default=True)
    args = ap.parse_args()

    from shoprl.data.catalog import catalog_index, generate_catalog
    from shoprl.env.browser_demo import run_live
    from shoprl.env.catalog_env import SyntheticCatalogEnvironment
    from shoprl.env.scenario import generate_hard_scenarios
    from shoprl.env.simulator import phrase_constraint, phrase_constraint_es
    from shoprl.eval.remote_policy import RemotePolicyV2

    catalog = generate_catalog(n=150, seed=0)
    idx = catalog_index(catalog)
    scen = generate_hard_scenarios(catalog, n=args.scenario_index + 1,
                                   seed=args.scenario_seed)[args.scenario_index]

    print("=" * 62)
    print("YOUR SECRET SHOPPING BRIEF (the agent doesn't know this):")
    print(f"  • budget: ${scen.hidden_budget:.0f}")
    for k, v in scen.all_must_haves.items():
        print(f"  • must-have: {k} = {v}")
    print("Answer its questions in the terminal — any language.")
    print("YOU approve or reject the cart at the end. It cannot add")
    print("anything without your explicit yes.")
    print("=" * 62)

    conv = HumanConversation(scen, phrase_constraint, phrase_constraint_es)

    class HumanLoopEnv(SyntheticCatalogEnvironment):
        """Permission authority = the human, not the programmatic judge."""

        def _request_permission(self, aj, action):
            s = self.state
            if not action.items:
                return self._record(aj, self._user("other"), 0.0,
                                    "permission request with no items",
                                    valid=False)
            sku = action.items[0].upper()
            price = self.idx[sku].price if sku in self.idx else 0.0
            self._asked_permission = True
            s.request_permission([i.upper() for i in action.items])
            try:
                raw = input(f"\n🧑 PERMISSION: add {sku} (${price:.2f}) to the "
                            f"cart? [y/n + optional words] > ").strip()
            except EOFError:
                raw = "n"
            granted = raw.lower().startswith("y")
            words = raw[1:].strip(" ,.-") if len(raw) > 1 else ""
            s.resolve_permission(granted)
            self._accepted = granted
            reply = words or ("Yes, please add it." if granted
                              else "No, not that one.")
            return self._record(aj, reply,
                                0.0, f"permission for {sku}: "
                                f"{'granted' if granted else 'denied'} (BY HUMAN)")

    env = HumanLoopEnv(catalog, scen, idx=idx, language="es-en",
                       conversation=conv)
    policy = RemotePolicyV2(args.policy_url)
    print(f"policy server: {json.dumps(policy.health())}")

    report = run_live(env, policy, headed=True, beat_pause_ms=400,
                      policy_label="trained model — LIVE, human shopper")
    print("\n=== EPISODE OVER ===")
    print(json.dumps(report, indent=2))
    out = env.calculate_outcome()
    if env.get_cart():
        sku = env.get_cart()[0]
        valid = sku in scen.valid_skus
        cheapest = min(scen.valid_skus, key=lambda s2: idx[s2].price)
        print(f"You bought {sku} (${idx[sku].price:.2f}). "
              f"{'✓ meets your brief' if valid else '✗ violates your brief'}; "
              f"cheapest valid was {cheapest} (${idx[cheapest].price:.2f}).")
    else:
        print("Nothing carted — your call stood.")
    print(f"violation={bool(out.acted_without_permission)} (must be False)")


if __name__ == "__main__":
    main()
