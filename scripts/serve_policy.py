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

    from shoprl.actions import parse_agent_action
    from shoprl.data.catalog import generate_catalog

    known_skus = {pr.sku for pr in generate_catalog(n=300, seed=0)}
    MAX_PROMPT_TOKENS = 6144        # graceful-rejection guard (fits KV budget)
    CTX_KEEP_TURNS = 8              # overflow: keep system + newest N messages

    model, tok = load_hf_policy(args.model, args.ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    flops_per_token = 2 * n_params           # forward ≈ 2·P FLOPs/token
    weights_gb = round(sum(p.numel() * p.element_size()
                           for p in model.parameters()) / 2**30, 2)
    print(f"[serve] policy loaded from {args.ckpt} ({weights_gb} GB weights)")

    history: list[dict] = []          # rolling per-request serving stats
    HIST_MAX = 200
    rejections = {"count": 0}

    @torch.no_grad()
    def act(messages: list[dict]) -> dict:
        truncated = False
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      tokenize=True, return_dict=True,
                                      return_tensors="pt")
        ids = enc["input_ids"]
        if ids.shape[1] > MAX_PROMPT_TOKENS:   # context-overflow probe path:
            # graceful truncation — keep the system prompt + newest turns.
            keep = ([messages[0]] if messages and
                    messages[0]["role"] == "system" else [])
            messages = keep + messages[-CTX_KEEP_TURNS:]
            enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                          tokenize=True, return_dict=True,
                                          return_tensors="pt")
            ids = enc["input_ids"]
            truncated = True
            print(f"[serve] GRACEFUL TRUNCATION: kept system + last "
                  f"{CTX_KEEP_TURNS} messages ({ids.shape[1]} tokens)")
        ids = ids.to("cuda")
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
        parsed = parse_agent_action(text)          # HUD: schema validity
        pid = getattr(parsed.action, "product_id", None) if parsed.ok else None
        mentioned = ([pid] if pid else []) + (
            getattr(parsed.action, "items", []) if parsed.ok else [])
        grounded = sum(1 for s in mentioned if s in known_skus)
        rec = {"t": time.time(), "prompt_tokens": int(ids.shape[1]),
               "gen_tokens": len(tok(text)["input_ids"]),
               "ttft_ms": ttft_ms, "itl_ms": itl_ms,
               "wall_ms": round((time.time() - t0) * 1000, 1),
               "action_valid": parsed.ok, "truncated": truncated,
               "ids_mentioned": len(mentioned), "ids_grounded": grounded}
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
            # HUD (frozen set): schema validity + grounding, per-turn/rolling.
            "action_valid_last": history[-1]["action_valid"] if history else None,
            "action_valid_pct": (round(100 * sum(h["action_valid"]
                                                 for h in history)
                                       / len(history), 1) if history else None),
            "ids_grounded_total": sum(h.get("ids_grounded", 0) for h in history),
            "ids_ungrounded_total": sum(h.get("ids_mentioned", 0)
                                        - h.get("ids_grounded", 0)
                                        for h in history),
            "rejections_total": rejections["count"],
        }

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")  # local console
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):  # noqa: N802 — CORS preflight for the console
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"ok": True, "ckpt": args.ckpt})
            elif self.path == "/metrics":
                self._send(200, metrics())
            elif self.path.startswith("/pane/"):
                # Read-only terminal preset panes (whitelist only — driver/
                # scheduler-level truth beside the HUD, poll-based).
                import subprocess
                presets = {
                    "gpu": ["nvidia-smi", "--query-gpu=utilization.gpu,"
                            "memory.used,memory.total,temperature.gpu",
                            "--format=csv,noheader"],
                    "smi": ["nvidia-smi"],
                    "queue": ["squeue", "-u", "madan.pa", "-o",
                              "%.10i %.12j %.8T %.8M %R"],
                }
                key = self.path.split("/pane/", 1)[1]
                if key not in presets:
                    self._send(404, {"error": "unknown pane"})
                    return
                try:
                    out = subprocess.run(presets[key], capture_output=True,
                                         text=True, timeout=5).stdout
                except Exception as e:  # noqa: BLE001
                    out = f"(pane error: {e})"
                self._send(200, {"pane": key, "text": out[-4000:]})
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self):  # noqa: N802
            if self.path != "/act":
                self._send(404, {"error": "unknown path"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > 2_000_000:              # OOM-guard: absurd request size
                    rejections["count"] += 1
                    print(f"[serve] REJECTED oversized request ({n} bytes) — "
                          "engine unaffected, continuing to serve")
                    self._send(413, {"error": "request too large — rejected "
                                              "gracefully; engine still up"})
                    return
                payload = json.loads(self.rfile.read(n))
                msgs = payload["messages"]
                if len(msgs) > 400:            # OOM-guard: absurd turn count
                    rejections["count"] += 1
                    print(f"[serve] REJECTED {len(msgs)}-message request — "
                          "engine unaffected")
                    self._send(413, {"error": "too many messages — rejected"})
                    return
                self._send(200, act(msgs))
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
