"""Generate a self-contained chat+shop demo page (no Playwright).

    python scripts/make_chat_demo.py --out demo/chat.html
    # then, with the 7B server tunnelled to :8765, just OPEN demo/chat.html
    # in any browser and type.

The page's own JavaScript calls the serve_policy.py server at --server-url
(default the SSH tunnel http://localhost:8765). Shopping is projected
client-side from what you type; the permission gate holds (no cart without an
approval click).
"""
import argparse
from pathlib import Path

from shoprl.data.catalog import generate_catalog
from shoprl.env.browser_demo import render_chat_demo_html


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo/chat.html")
    ap.add_argument("--server-url", default="http://localhost:8765")
    ap.add_argument("--catalog-n", type=int, default=150)
    ap.add_argument("--catalog-seed", type=int, default=0)
    args = ap.parse_args()

    catalog = generate_catalog(n=args.catalog_n, seed=args.catalog_seed)
    html = render_chat_demo_html(catalog, server_url=args.server_url)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({len(html):,} bytes, {len(catalog)} products, "
          f"server {args.server_url})")
    print(f"open it:  open {out}   (with the tunnel up on :8765)")


if __name__ == "__main__":
    main()
