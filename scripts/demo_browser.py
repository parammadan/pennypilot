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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--slow-mo", type=int, default=0,
                    help="ms of Playwright slow-motion per operation (headed demos)")
    ap.add_argument("--beat-pause", type=int, default=900)
    ap.add_argument("--screenshots", default=None)
    args = ap.parse_args()

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
