"""Run the PennyData platform service (ingest API + live console).

    python scripts/run_platform.py --root ~/pennydata --port 8770
    # console:  http://localhost:8770/
"""
import argparse
import threading
from pathlib import Path

from shoprl.platform.ingest import serve

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "pennydata"))
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--kafka", default=None,
                    help="broker list (e.g. localhost:9092) — consume "
                         "pennymart.events into the store alongside HTTP")
    ap.add_argument("--from-beginning", action="store_true",
                    help="replay the topic from offset 0 (rebuilds all views)")
    ap.add_argument("--consumers", type=int, default=1,
                    help="consumer threads in ONE group — Kafka splits the "
                         "topic's partitions across them (horizontal scaling)")
    args = ap.parse_args()
    if args.kafka:
        import time as _t
        from shoprl.platform.store import PlatformStore
        from shoprl.platform.streaming import consume_into_store
        store = PlatformStore(args.root)
        group = ("pennydata-replay-" + str(int(_t.time()))
                 if args.from_beginning else "pennydata")
        for _ in range(max(1, args.consumers)):
            threading.Thread(
                target=consume_into_store, args=(store, args.kafka),
                kwargs={"from_beginning": args.from_beginning,
                        "group_id": group}, daemon=True).start()
        print(f"[pennydata] {args.consumers} kafka consumer(s) on {args.kafka} "
              f"group={group} (from_beginning={args.from_beginning})")
    serve(args.root, args.port)
