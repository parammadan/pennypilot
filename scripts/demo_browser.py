"""Visible-Chromium replay of a saved PennyPilot trajectory (Stage 5, replay).

    # watch it live (opens a Chromium window):
    python scripts/demo_browser.py --transcript runs/demo/oracle_esen.json --headed

    # headless artifact run (screenshots only):
    python scripts/demo_browser.py --transcript runs/demo/oracle_esen.json \
        --screenshots runs/demo/shots

Demo only — a projection of recorded structured actions onto a simulated local
storefront. No real site, no real cart, nothing to buy.
"""
from __future__ import annotations

import argparse
import json

from shoprl.env.browser_demo import replay_transcript


class _WebShopWalkthrough:
    """Deterministic demo driver for the WebShop store (no model needed):
    search -> select cheapest -> request permission -> buy."""

    def __init__(self, env):
        self.env = env
        self.i = 0

    def reset(self, scenario=None, idx=None):
        self.i = 0

    def act(self, observation=""):
        self.i += 1
        if self.i == 1:
            return json.dumps({"action": "search", "query": "sunscreen SPF 50"})
        cands = self.env.get_candidates()
        target = cands[0].asin if cands else "B01AAAA005"
        price = cands[0].price if cands else 0.0
        steps = [
            {"action": "select_product", "product_id": target,
             "reason": "cheapest match for the instruction"},
            {"action": "request_cart_permission", "items": [target],
             "estimated_total": price},
            {"action": "add_to_cart", "product_id": target},
        ]
        return json.dumps(steps[min(self.i - 2, 2)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default=None)
    ap.add_argument("--live", action="store_true",
                    help="policy decides NOW (remote server or oracle) instead of replay")
    ap.add_argument("--policy-url", default=None,
                    help="act-server URL (scripts/serve_policy.py via ssh tunnel); "
                         "omit for the scripted oracle")
    ap.add_argument("--store", default="pennymart", choices=["pennymart", "webshop"])
    ap.add_argument("--instruction",
                    default="Necesito sunscreen SPF 50 under $15 para los niños.")
    ap.add_argument("--language", default="es-en", choices=["en", "es", "es-en"])
    ap.add_argument("--scenario-seed", type=int, default=11)
    ap.add_argument("--scenario-index", type=int, default=2)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--slow-mo", type=int, default=0,
                    help="ms of Playwright slow-motion per operation (headed demos)")
    ap.add_argument("--beat-pause", type=int, default=900)
    ap.add_argument("--screenshots", default=None)
    args = ap.parse_args()

    if args.live:
        from shoprl.data.catalog import catalog_index, generate_catalog
        from shoprl.env.browser_demo import run_live
        from shoprl.env.catalog_env import SyntheticCatalogEnvironment
        from shoprl.env.scenario import generate_hard_scenarios

        if args.store == "webshop":
            from shoprl.env.webshop_env import WebShopEnvironment
            env = WebShopEnvironment(instruction=args.instruction, max_turns=12)
            if args.policy_url:
                from shoprl.eval.remote_policy import RemotePolicyV2
                policy = RemotePolicyV2(args.policy_url)
                label = f"LIVE remote policy ({policy.health().get('ckpt', '?')})"
            else:
                policy = _WebShopWalkthrough(env)
                label = "LIVE scripted walkthrough"
            print(f"[demo] LIVE webshop — {label}")
            report = run_live(env, policy, headed=args.headed,
                              slow_mo=args.slow_mo,
                              screenshot_dir=args.screenshots,
                              beat_pause_ms=args.beat_pause, policy_label=label)
            print(json.dumps(report, indent=2))
            if not report["cart_ok"]:
                raise SystemExit("[demo] browser cart diverged from env state")
            return

        catalog = generate_catalog(n=150, seed=0)
        idx = catalog_index(catalog)
        scen = generate_hard_scenarios(catalog, n=args.scenario_index + 1,
                                       seed=args.scenario_seed)[args.scenario_index]
        env = SyntheticCatalogEnvironment(catalog, scen, idx=idx,
                                          language=args.language)
        if args.policy_url:
            from shoprl.eval.remote_policy import RemotePolicyV2
            policy = RemotePolicyV2(args.policy_url)
            label = f"LIVE remote policy ({policy.health().get('ckpt', '?')})"
        else:
            from shoprl.eval.v2_policies import OracleGoodV2
            policy = OracleGoodV2()
            label = "LIVE oracle (scripted ceiling)"
        print(f"[demo] LIVE {scen.scenario_id} ({args.language}) — {label}")
        report = run_live(env, policy, headed=args.headed, slow_mo=args.slow_mo,
                          screenshot_dir=args.screenshots,
                          beat_pause_ms=args.beat_pause, policy_label=label)
        print(json.dumps(report, indent=2))
        if args.headed:
            print(">>> HUMAN CAPTURE: live episode just ran — the model decided "
                  "each action in real time. Screen-record a rerun for video. <<<")
        if not report["cart_ok"]:
            raise SystemExit("[demo] browser cart diverged from env state")
        return

    if not args.transcript:
        raise SystemExit("need --transcript (replay) or --live")
    with open(args.transcript) as f:
        bundle = json.load(f)
    print(f"[demo] replaying {bundle['record']['scenario_id']} "
          f"({bundle['language']}, policy: {bundle.get('policy', '?')})")
    report = replay_transcript(bundle, headed=args.headed, slow_mo=args.slow_mo,
                               screenshot_dir=args.screenshots,
                               beat_pause_ms=args.beat_pause)
    print(json.dumps(report, indent=2))
    if args.headed:
        print(">>> HUMAN CAPTURE: the Chromium window just walked the full "
            "episode (search → results → selection → permission modal → "
            "simulated add). Re-run and screen-record if you want video. <<<")
    if not report["cart_ok"]:
        raise SystemExit("[demo] cart badge mismatch vs recorded trajectory")


if __name__ == "__main__":
    main()
