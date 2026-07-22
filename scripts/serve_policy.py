"""Policy act-server (GPU) — serves next-action generation for the LIVE demo.

    python scripts/serve_policy.py --ckpt /scratch/madan.pa/pennypilot/rloo50_v2/policy \
        --port 8765

    # from the laptop, tunnel through the login node (node printed on start):
    ssh -L 8765:<node>:8765 explorer
    # then the demo talks to http://localhost:8765

Protocol: POST /act  {"messages": [{"role","content"}...]} -> {"text": "..."}
          GET  /health -> {"ok": true, "ckpt": ...}
Stdlib http.server on purpose (no extra deps on the cluster); single-threaded
is correct here — the demo is one conversation at a time.
"""
from __future__ import annotations

import argparse
import json
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    import torch

    from shoprl.profiling.bench_common import load_hf_policy

    model, tok = load_hf_policy(args.model, args.ckpt)
    print(f"[serve] policy loaded from {args.ckpt}")

    @torch.no_grad()
    def act(messages: list[dict]) -> str:
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      tokenize=True, return_dict=True,
                                      return_tensors="pt")
        ids = enc["input_ids"].to("cuda")
        out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"ok": True, "ckpt": args.ckpt})
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self):  # noqa: N802
            if self.path != "/act":
                self._send(404, {"error": "unknown path"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n))
                text = act(payload["messages"])
                self._send(200, {"text": text})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *a):  # quiet
            print(f"[serve] {self.address_string()} {fmt % a}")

    node = socket.gethostname()
    print(f"[serve] listening on {node}:{args.port}")
    print(f">>> tunnel from laptop:  ssh -L {args.port}:{node}:{args.port} explorer")
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
