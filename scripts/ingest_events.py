"""Ingest a platform-schema events JSONL file (e.g. a behavior-campaign
output) into a PennyData store. Idempotent: re-ingesting the same file
cannot double-count (event_id dedup).

    python scripts/ingest_events.py --root ~/pennydata --file campaign.jsonl
"""
import argparse
import json

from shoprl.platform.store import PlatformStore

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--file", required=True)
    args = ap.parse_args()
    store = PlatformStore(args.root)
    n = dup = bad = 0
    for line in open(args.file):
        try:
            kind = store.ingest(json.loads(line))
        except Exception as e:
            bad += 1
            print(f"[ingest] BAD EVENT kept out ({e}): {line[:120]}")
            continue
        if kind == "duplicate":
            dup += 1
        else:
            n += 1
    print(f"[ingest] {n} ingested, {dup} duplicates skipped, {bad} malformed")
