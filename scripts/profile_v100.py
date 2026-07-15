"""VRAM / max-sequence-length profiler for Pennywise on the V100.

Volta has no FlashAttention, so attention memory is quadratic in sequence
length. This sweeps (batch x seq_len) and records the peak VRAM of a realistic
training step for full fine-tune (and LoRA), so max_turns / max_new_tokens /
num_samples can be set from measurement, not guesswork.

A "step" here is: reference forward (no_grad, full-FT only) + policy forward with
labels -> loss -> backward -> optimizer.step(). That mirrors what the RLOO loop
does per update, so the peak is representative (not just a bare forward).

    python scripts/profile_v100.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --methods full lora --seq-lens 512 1024 1536 2048 3072 --batches 1 4 8 \
        --out runs/profile_v100.json
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from shoprl.train.build import build_model

GB = 1e9


def _is_oom(err: Exception) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


def profile_step(bm, optimizer, batch: int, seq_len: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tok = bm.tokenizer
    vocab = getattr(bm.policy.config, "vocab_size", len(tok))
    ids = torch.randint(0, vocab, (batch, seq_len), device=bm.device)
    attn = torch.ones_like(ids)
    try:
        if bm.reference is not None:                 # RLOO needs a ref forward
            with torch.no_grad():
                bm.reference(input_ids=ids, attention_mask=attn)
        out = bm.policy(input_ids=ids, attention_mask=attn, labels=ids)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / GB
        return {"batch": batch, "seq_len": seq_len, "tokens": batch * seq_len,
                "ok": True, "peak_gb": round(peak, 2)}
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if not _is_oom(e):
            raise
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return {"batch": batch, "seq_len": seq_len, "tokens": batch * seq_len,
                "ok": False, "peak_gb": None, "note": "OOM"}


def profile_method(model: str, method: str, seq_lens, batches, dtype, cap_gb) -> dict:
    bm = build_model(model, method=method, dtype=dtype, device="cuda",
                     grad_checkpointing=True)
    optimizer = torch.optim.AdamW(
        (p for p in bm.policy.parameters() if p.requires_grad), lr=1e-6)
    # Fixed (config-independent) footprint: weights + optimizer state, measured
    # after one tiny step allocates the optimizer moments.
    warm = profile_step(bm, optimizer, 1, 64)
    fixed_gb = warm.get("peak_gb")

    results = []
    configs = sorted({(b, s) for b in batches for s in seq_lens},
                     key=lambda bs: bs[0] * bs[1])
    oom_floor = None  # smallest token count that OOM'd
    for batch, seq in configs:
        if oom_floor is not None and batch * seq >= oom_floor:
            results.append({"batch": batch, "seq_len": seq, "tokens": batch * seq,
                            "ok": False, "peak_gb": None, "note": "skipped (> OOM floor)"})
            continue
        r = profile_step(bm, optimizer, batch, seq)
        if r["ok"] and r["peak_gb"] > cap_gb:
            r["ok"] = False
            r["note"] = f"over cap {cap_gb}GB"
        results.append(r)
        if not r["ok"] and r.get("note", "").startswith(("OOM", "over cap")):
            oom_floor = batch * seq
    del bm, optimizer
    torch.cuda.empty_cache()
    return {"method": method, "fixed_gb": fixed_gb, "trainable_frac": None,
            "results": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--methods", nargs="+", default=["full", "lora"])
    ap.add_argument("--seq-lens", nargs="+", type=int,
                    default=[512, 1024, 1536, 2048, 3072])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8])
    ap.add_argument("--cap-gb", type=float, default=30.0)  # leave headroom on 32
    ap.add_argument("--out", default="runs/profile_v100.json")
    args = ap.parse_args()

    dev = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.mem_get_info()[1] / GB
    print(f"[profile] {dev} total={total_gb:.1f}GB cap={args.cap_gb}GB "
          f"model={args.model}")

    report = {"device": dev, "total_gb": round(total_gb, 2), "cap_gb": args.cap_gb,
              "model": args.model, "dtype": "float16", "methods": []}
    for method in args.methods:
        print(f"\n[profile] method={method}")
        m = profile_method(args.model, method, args.seq_lens, args.batches,
                           torch.float16, args.cap_gb)
        report["methods"].append(m)
        print(f"  fixed footprint (weights+optim+ctx): {m['fixed_gb']} GB")
        for r in m["results"]:
            flag = f"{r['peak_gb']}GB" if r["ok"] else f"FAIL ({r.get('note','')})"
            print(f"  batch={r['batch']:>2} seq={r['seq_len']:>5} "
                  f"tokens={r['tokens']:>6} -> {flag}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[profile] wrote {args.out}")


if __name__ == "__main__":
    main()
