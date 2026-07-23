"""HUMAN-IN-THE-LOOP demo — YOU are the shopper, chatting IN THE BROWSER.

    python scripts/demo_human.py                     # browser chat UI (default)
    python scripts/demo_human.py --no-browser        # pure terminal chat

Default mode: PennyMart opens with a chat panel — type in the box at the
bottom left, press Enter/Send. The input placeholder always tells you what
the agent is waiting for. When it asks permission, the modal's Approve /
"Not yet" buttons are YOUR buttons — the model cannot cart without your click.

Ground-truth note: your secret brief (shown at start) is what the store's
valid/cheapest sets follow — phrase your answers however you like (any
language), but keep the NUMBERS near the brief or the store won't match your
words. Your permission click is the real authority in this mode.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def brief_text(scen) -> str:
    lines = [f"budget ${scen.hidden_budget:.0f}"]
    lines += [f"{k} = {v}" for k, v in scen.all_must_haves.items()]
    return "; ".join(lines)


def _agent_say(text: str, chat: bool) -> str:
    """Chat-face mode shows the friendly prose to the shopper and hides the raw
    JSON action line (the JSON drives the store projection, not the bubble)."""
    if not chat:
        return text
    prose = text.split("{", 1)[0].strip()
    return prose or "(one moment…)"


# ---------------- terminal mode pieces (unchanged behaviour) ----------------
class TerminalHuman:
    def __init__(self, phrase):
        self._phrase = phrase

    def _ask(self, prompt, default):
        try:
            text = input(f"\n🧑 {prompt}\n   [Enter = “{default}”] > ").strip()
        except EOFError:
            text = ""
        return text or default

    def utter(self, intent, scenario, accepted=None):
        if intent == "greet":
            return self._ask("Your opening request (don't reveal specifics yet):",
                             "Necesito una laptop, can you help?")
        if intent == "budget":
            return self._ask(f"It asked your budget (brief: "
                             f"${scenario.hidden_budget:.0f}):",
                             f"My budget is about ${scenario.hidden_budget:.0f}.")
        if intent == "permission":
            return ("Yes, please add it." if accepted
                    else "No, that one doesn't work for me.")
        return self._ask("Your reply:", "Could you help me find the right one?")

    def utter_constraint(self, key, value):
        return self._ask(f"It asked about requirements (brief: {key}={value}):",
                         self._phrase(key, value))


# ---------------- browser-chat mode pieces ----------------------------------
class BrowserHuman:
    """ConversationModel seam wired to the in-page input bar."""

    def __init__(self, page, timeout_ms: int = 300_000):
        self.page = page
        self.timeout_ms = timeout_ms

    def _wait(self, hint: str, default: str) -> str:
        self.page.evaluate(f"pennymart.hint({json.dumps(hint)})")
        self.page.evaluate("window.__human = null")
        try:
            self.page.wait_for_function("window.__human !== null",
                                        timeout=self.timeout_ms)
            v = self.page.evaluate("window.__human")
            self.page.evaluate("window.__human = null")
            return (v or "").strip() or default
        except Exception:
            return default

    def utter(self, intent, scenario, accepted=None):
        if intent == "greet":
            return self._wait("say hi + what you're shopping for (any language; "
                              "no specifics yet)…",
                              "Necesito una laptop, can you help?")
        if intent == "budget":
            return self._wait(f"tell it your budget (your brief: "
                              f"${scenario.hidden_budget:.0f})…",
                              f"My budget is about ${scenario.hidden_budget:.0f}.")
        if intent == "permission":
            return ("Yes, please add it." if accepted else "No, not that one.")
        return self._wait("your reply…", "Could you help me find the right one?")

    def utter_constraint(self, key, value):
        return self._wait(f"tell it this requirement (brief: {key} = {value})…",
                          f"It needs {key} = {value}.")


def run_browser_chat(args) -> None:
    from playwright.sync_api import sync_playwright

    from shoprl.actions import parse_agent_action
    from shoprl.data.catalog import catalog_index, generate_catalog
    from shoprl.env.browser_demo import _project_action, render_storefront_html
    from shoprl.env.catalog_env import SyntheticCatalogEnvironment
    from shoprl.env.scenario import generate_hard_scenarios
    from shoprl.eval.remote_policy import RemotePolicyV2

    catalog = generate_catalog(n=150, seed=0)
    idx = catalog_index(catalog)
    scen = generate_hard_scenarios(catalog, n=args.scenario_index + 1,
                                   seed=args.scenario_seed)[args.scenario_index]
    html = Path(tempfile.mkdtemp(prefix="pennymart-human-")) / "pennymart.html"
    html.write_text(render_storefront_html(catalog))

    if args.chat:
        from shoprl.data.prompts_v2 import SYSTEM_PROMPT_CHAT, SYSTEM_PROMPT_CHAT_MIN
        system = SYSTEM_PROMPT_CHAT_MIN if args.chat_min else SYSTEM_PROMPT_CHAT
        policy = RemotePolicyV2(args.policy_url, system=system)
    else:
        policy = RemotePolicyV2(args.policy_url)
    print("policy server:", json.dumps(policy.health()))
    print(f"SUGGESTED BRIEF (optional — say whatever you want!): {brief_text(scen)}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(f"file://{html}")
        human = BrowserHuman(page)

        class UIHumanEnv(SyntheticCatalogEnvironment):
            """Human-truth mode: the store believes YOUR WORDS (parsed by the
            language layer), not the brief; permission = YOUR click."""

            def _filter_products(self, k=None):
                s = self.state
                out = []
                for p2 in self.catalog:
                    if s.budget_total and p2.price > s.budget_total:
                        continue
                    ok = True
                    for key, v in s.hard_constraints.items():
                        if key == "brand":
                            ok = str(p2.brand).lower() == str(v).lower()
                        elif key == "min_ram":
                            ok = p2.ram_gb >= float(v)
                        elif key == "min_battery":
                            ok = p2.battery_hrs >= float(v)
                        elif key == "max_weight":
                            ok = p2.weight_lbs <= float(v)
                        if not ok:
                            break
                    if ok:
                        out.append(p2)
                out.sort(key=lambda p2: p2.price)
                return out[:k] if k is not None else out

            def _ask(self, aj, question):
                before = self._consistent_count()
                user = human._wait(f"🤖 {question}", "hmm, whatever you think")
                self.state.observe_user_message(user)   # YOUR words = the truth
                gain = self._gain(before, self._consistent_count())
                understood = (self.state.normalized_english_intent
                              or "(nothing specific understood)")
                return self._record(aj, user, gain, f"understood {understood}")

            def _request_permission(self, aj, action):
                s = self.state
                if not action.items:
                    return self._record(aj, "…", 0.0,
                                        "permission request with no items",
                                        valid=False)
                sku = action.items[0].upper()
                self._asked_permission = True
                s.request_permission([i.upper() for i in action.items])
                page.evaluate("pennymart.permission(%s, %s, %s, %s)" % (
                    json.dumps(action.items),
                    json.dumps(action.estimated_total),
                    json.dumps(s.estimated_savings),
                    json.dumps("YOUR decision — click Approve or Not yet "
                               "(or type yes/no)")))
                reply = human._wait("…or type your answer here…", "__no__")
                page.evaluate("pennymart.closeModal()")
                low = reply.lower()
                granted = (reply == "__yes__" or
                           low.startswith(("y", "yes", "si", "sí", "ok")))
                words = ("" if reply in ("__yes__", "__no__") else reply)
                s.resolve_permission(granted)
                self._accepted = granted
                final = words or ("Yes, please add it." if granted
                                  else "No, not that one.")
                return self._record(aj, final, 0.0,
                                    f"permission for {sku}: "
                                    f"{'granted' if granted else 'denied'} "
                                    "(BY YOU)")

        env = UIHumanEnv(catalog, scen, idx=idx, language="es-en",
                         conversation=human)
        page.evaluate(f"pennymart.demoHint({json.dumps(brief_text(scen))})")

        opener = env.reset()          # waits for YOUR first message in the box
        policy.reset()
        page.evaluate(f"pennymart.bubble('user', {json.dumps(opener)})")
        obs = env.observe()
        done = False
        steps = 0
        while not done and steps < (40 if args.chat else 15):
            action_text = policy.act(obs)
            exec_text = action_text
            if args.chat and "{" not in action_text:
                # rehearsal-trained policies answer non-shopping turns as pure
                # prose (no action) — route through ask_user so the env waits
                # for the human instead of flagging an invalid action
                exec_text = action_text + " " + json.dumps(
                    {"action": "ask_user", "question": action_text})
            step = env.execute_text(exec_text)
            note = "" if exec_text != action_text else step.note
            page.evaluate(f"pennymart.bubble('agent', "
                          f"{json.dumps(_agent_say(action_text, args.chat))}, "
                          f"{json.dumps(note)})")
            r = parse_agent_action(exec_text)
            if r.ok and r.action.action != "request_cart_permission":
                _project_action(page, r, step.note, step.observation,
                                env.state.estimated_savings,
                                lambda name: None, 300)
            if step.observation:
                page.evaluate(f"pennymart.bubble('user', "
                              f"{json.dumps(step.observation[:400])})")
            obs = step.observation
            done = step.done
            steps += 1

        out = env.calculate_outcome()
        verdict = "Nothing carted — your call stood."
        if env.get_cart():
            sku = env.get_cart()[0]
            stated = env._filter_products()      # valid per what YOU SAID
            valid = any(p.sku == sku for p in stated)
            cheapest = stated[0] if stated else None
            verdict = (f"You bought {sku} (${idx[sku].price:.2f}) — "
                       f"{'✓ fits what you told it' if valid else '✗ misses what you told it'}"
                       + (f"; cheapest match for your stated needs was "
                          f"{cheapest.sku} (${cheapest.price:.2f})."
                          if cheapest else "."))
        page.evaluate(f"pennymart.bubble('user', {json.dumps('🏁 ' + verdict)}, "
                      f"'violation={bool(out.acted_without_permission)}')")
        print("\n=== EPISODE OVER ===\n" + verdict)
        print(f"violation={bool(out.acted_without_permission)} (must be False)")
        page.evaluate("pennymart.hint('episode over — close the window when done')")
        try:
            page.wait_for_function("window.__closed === true", timeout=600_000)
        except Exception:
            pass
        browser.close()


def run_terminal_chat(args) -> None:
    from shoprl.data.catalog import catalog_index, generate_catalog
    from shoprl.env.catalog_env import SyntheticCatalogEnvironment
    from shoprl.env.scenario import generate_hard_scenarios
    from shoprl.env.simulator import phrase_constraint
    from shoprl.eval.remote_policy import RemotePolicyV2

    catalog = generate_catalog(n=150, seed=0)
    idx = catalog_index(catalog)
    scen = generate_hard_scenarios(catalog, n=args.scenario_index + 1,
                                   seed=args.scenario_seed)[args.scenario_index]
    print("=" * 62)
    print(f"YOUR SECRET BRIEF: {brief_text(scen)}")
    print("=" * 62)
    conv = TerminalHuman(phrase_constraint)

    class HumanLoopEnv(SyntheticCatalogEnvironment):
        def _request_permission(self, aj, action):
            s = self.state
            if not action.items:
                return self._record(aj, "…", 0.0, "no items", valid=False)
            sku = action.items[0].upper()
            price = self.idx[sku].price if sku in self.idx else 0.0
            self._asked_permission = True
            s.request_permission([i.upper() for i in action.items])
            try:
                raw = input(f"\n🧑 PERMISSION: add {sku} (${price:.2f})? "
                            f"[y/n + words] > ").strip()
            except EOFError:
                raw = "n"
            granted = raw.lower().startswith("y")
            s.resolve_permission(granted)
            self._accepted = granted
            words = raw[1:].strip(" ,.-") if len(raw) > 1 else ""
            return self._record(aj, words or ("Yes, add it." if granted
                                              else "No, not that one."),
                                0.0, f"permission: "
                                f"{'granted' if granted else 'denied'} (BY YOU)")

    env = HumanLoopEnv(catalog, scen, idx=idx, language="es-en",
                       conversation=conv)
    if args.chat:
        from shoprl.data.prompts_v2 import SYSTEM_PROMPT_CHAT, SYSTEM_PROMPT_CHAT_MIN
        system = SYSTEM_PROMPT_CHAT_MIN if args.chat_min else SYSTEM_PROMPT_CHAT
        policy = RemotePolicyV2(args.policy_url, system=system)
    else:
        policy = RemotePolicyV2(args.policy_url)
    print("policy server:", json.dumps(policy.health()))
    opener = env.reset()
    policy.reset()
    print(f"\n🧑 you: {opener}")
    obs = env.observe()
    done = False
    steps = 0
    while not done and steps < (40 if args.chat else 15):
        action_text = policy.act(obs)
        print(f"🤖 agent: {_agent_say(action_text, args.chat)}")
        exec_text = action_text
        if args.chat and "{" not in action_text:
            exec_text = action_text + " " + json.dumps(
                {"action": "ask_user", "question": action_text})
        step = env.execute_text(exec_text)
        if step.observation:
            print(f"🏪 store/you: {step.observation}")
        obs = step.observation
        done = step.done
        steps += 1
    out = env.calculate_outcome()
    print("\n=== EPISODE OVER ===")
    if env.get_cart():
        sku = env.get_cart()[0]
        cheapest = min(scen.valid_skus, key=lambda x: idx[x].price)
        print(f"You bought {sku} (${idx[sku].price:.2f}); "
              f"{'✓ fits brief' if sku in scen.valid_skus else '✗ off-brief'}; "
              f"cheapest valid {cheapest} (${idx[cheapest].price:.2f}).")
    else:
        print("Nothing carted — your call stood.")
    print(f"violation={bool(out.acted_without_permission)} (must be False)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-url", default="http://localhost:8765")
    ap.add_argument("--scenario-seed", type=int, default=11)
    ap.add_argument("--scenario-index", type=int, default=4)
    ap.add_argument("--no-browser", action="store_true",
                    help="pure terminal chat instead of the browser UI")
    ap.add_argument("--chat", action="store_true",
                    help="talk to the bigger Qwen2.5-7B-Instruct chat-face "
                         "(SYSTEM_PROMPT_CHAT): natural prose + store actions. "
                         "Point --policy-url at a serve_7b.sh server.")
    ap.add_argument("--chat-min", action="store_true",
                    help="with --chat, use SYSTEM_PROMPT_CHAT_MIN — the prompt "
                         "the H3 7B arms were trained/evaluated on; use when "
                         "serving a trained adapter (e.g. rl7b_B) not the base")
    args = ap.parse_args()
    if args.no_browser:
        run_terminal_chat(args)
    else:
        run_browser_chat(args)


if __name__ == "__main__":
    main()
