"""Policy act-server (GPU) — serves next-action generation for the LIVE demo.

    python scripts/serve_policy.py --ckpt /scratch/madan.pa/pennypilot/rloo50_v2/policy \
        --port 8765

    # from the laptop, tunnel through the login node (node printed on start):
    ssh -L 8765:<node>:8765 explorer
    # then the demo talks to http://localhost:8765

Protocol: POST /act  {"messages": [...]} -> {"text","ttft_ms","itl_ms","gen_tokens"}
          GET  /health  -> {"ok": true, "ckpt": ...}
          GET  /metrics -> rolling serving stats (TTFT p50/p95, ITL, tok/s,
                           request count) + live GPU telemetry (util%, mem GB,
                           weights/optimizer-free memory ledger inputs)
Stdlib http.server on purpose (no frameworks — the console polls /metrics).
TTFT/ITL are measured with a real token streamer on the actual generate call.
"""
from __future__ import annotations

import argparse
import json
import socket
import statistics
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--peak-flops", type=float, default=125e12,
                    help="hardware fp16 peak for MFU (V100 tensor: 125e12; "
                         "H200: ~989e12)")
    args = ap.parse_args()

    import torch

    from shoprl.profiling.bench_common import load_hf_policy

    from transformers import TextIteratorStreamer

    model, tok = load_hf_policy(args.model, args.ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    flops_per_token = 2 * n_params           # forward ≈ 2·P FLOPs/token
    weights_gb = round(sum(p.numel() * p.element_size()
                           for p in model.parameters()) / 2**30, 2)
    print(f"[serve] policy loaded from {args.ckpt} ({weights_gb} GB weights)")

    history: list[dict] = []          # rolling per-request serving stats
    HIST_MAX = 200

    @torch.no_grad()
    def act(messages: list[dict]) -> dict:
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      tokenize=True, return_dict=True,
                                      return_tensors="pt")
        ids = enc["input_ids"].to("cuda")
        streamer = TextIteratorStreamer(tok, skip_prompt=True,
                                        skip_special_tokens=True)
        t0 = time.time()
        th = threading.Thread(target=model.generate, kwargs=dict(
            inputs=ids, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id, streamer=streamer))
        th.start()
        chunks, stamps = [], []
        for piece in streamer:
            stamps.append(time.time())
            chunks.append(piece)
        th.join()
        text = "".join(chunks).strip()
        ttft_ms = round((stamps[0] - t0) * 1000, 1) if stamps else None
        itl_ms = (round(((stamps[-1] - stamps[0]) / max(len(stamps) - 1, 1))
                        * 1000, 2) if len(stamps) > 1 else None)
        rec = {"t": time.time(), "prompt_tokens": int(ids.shape[1]),
               "gen_tokens": len(tok(text)["input_ids"]),
               "ttft_ms": ttft_ms, "itl_ms": itl_ms,
               "wall_ms": round((time.time() - t0) * 1000, 1)}
        history.append(rec)
        del history[:-HIST_MAX]
        return {"text": text, **rec}

    def metrics() -> dict:
        from shoprl.profiling.capture import default_sample_fn
        try:
            util, mem_gb = default_sample_fn()
        except Exception:
            util, mem_gb = None, None
        ttfts = [h["ttft_ms"] for h in history if h["ttft_ms"]]
        itls = [h["itl_ms"] for h in history if h["itl_ms"]]
        recent = [h for h in history if h["t"] > time.time() - 60]
        return {
            "requests_total": len(history),
            "requests_last_60s": len(recent),
            "ttft_ms_p50": round(statistics.median(ttfts), 1) if ttfts else None,
            "ttft_ms_p95": (round(sorted(ttfts)[int(0.95 * (len(ttfts) - 1))], 1)
                            if ttfts else None),
            "itl_ms_mean": round(statistics.mean(itls), 2) if itls else None,
            "tokens_per_sec_60s": round(sum(h["gen_tokens"] for h in recent)
                                        / 60.0, 1),
            "gpu_util_pct": util,
            "gpu_mem_gb": mem_gb,
            "weights_gb": weights_gb,
            "torch_alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
            # MFU, honestly split: decode from rolling gen tok/s; prefill from
            # the LAST request's prompt_tokens/TTFT. Decode MFU is inherently
            # low (bandwidth-bound) — the console displays that note verbatim.
            "mfu_decode_pct": (round(100 * (sum(h["gen_tokens"] for h in recent)
                                            / 60.0) * flops_per_token
                                     / args.peak_flops, 3) if recent else None),
            "mfu_prefill_last_pct": (round(100 * history[-1]["prompt_tokens"]
                                           * flops_per_token
                                           / ((history[-1]["ttft_ms"] or 1e9)
                                              / 1000) / args.peak_flops, 2)
                                     if history and history[-1]["ttft_ms"]
                                     else None),
            "peak_flops": args.peak_flops,
            # Honestly sourceable only where APC runs (Ampere+): on Volta this
            # stays a verdict string, never an approximation.
            "prefix_cache": "hardware-gated on Volta (CHALLENGES #26); live "
                            "hit-rate wires up on the H200 leg",
        }

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
            elif self.path == "/metrics":
                self._send(200, metrics())
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self):  # noqa: N802
            if self.path != "/act":
                self._send(404, {"error": "unknown path"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n))
                self._send(200, act(payload["messages"]))
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
