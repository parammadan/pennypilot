"""Backfill PennyData from existing capture bundles (demos/live_sessions/*/
transcript.json) so the platform isn't empty before the first live session.

    python scripts/platform_load.py --root ~/pennydata --bundles demos/live_sessions
"""
import argparse
import json
from pathlib import Path

from shoprl.platform.store import PlatformStore


def load_bundle(store: PlatformStore, path: Path) -> int:
    b = json.loads(path.read_text())
    sid = path.parent.name
    n = 0
    store.ingest({"kind": "episode_start", "session_id": sid,
                  "label": b.get("label") or "", "policy": b.get("policy", {}),
                  "brief": b.get("brief", "")})
    n += 1
    for t in b.get("turns", []):
        store.ingest({"kind": "turn", "session_id": sid, "i": t["turn"],
                      "agent": t["agent"], "observation": t.get("observation") or "",
                      "note": t.get("note") or ""})
        n += 1
        if t.get("feedback"):
            store.ingest({"kind": "feedback", "session_id": sid, "i": t["turn"],
                          "vote": t["feedback"]})
            n += 1
    store.ingest({"kind": "episode_end", "session_id": sid,
                  "verdict": b.get("verdict", ""),
                  "violation": bool(b.get("violation")),
                  "cart": b.get("cart", [])})
    return n + 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--bundles", default="demos/live_sessions")
    args = ap.parse_args()
    store = PlatformStore(args.root)
    total = 0
    for p in sorted(Path(args.bundles).glob("*/transcript.json")):
        total += load_bundle(store, p)
        print(f"[load] {p.parent.name}")
    print(f"[load] {total} events ingested -> {store.root}")
