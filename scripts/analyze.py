"""Run the PennyData automated analysis and write the report.

    python scripts/analyze.py --root ~/pennydata [--out report.md]
"""
import argparse
import json
from pathlib import Path

from shoprl.platform.report import analyze, to_markdown
from shoprl.platform.store import PlatformStore

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None, help="markdown path (default stdout)")
    ap.add_argument("--min-support", type=int, default=30)
    args = ap.parse_args()
    r = analyze(PlatformStore(args.root), min_support=args.min_support)
    md = to_markdown(r)
    if args.out:
        Path(args.out).write_text(md)
        Path(args.out).with_suffix(".json").write_text(json.dumps(r, indent=1))
        print(f"[analyze] report -> {args.out} (+ .json)")
    else:
        print(md)
