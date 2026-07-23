"""Run the PennyData platform service (ingest API + live console).

    python scripts/run_platform.py --root ~/pennydata --port 8770
    # console:  http://localhost:8770/
"""
import argparse
from pathlib import Path

from shoprl.platform.ingest import serve

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "pennydata"))
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    serve(args.root, args.port)
