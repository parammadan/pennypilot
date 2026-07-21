"""Training-side building blocks for Pennywise (Phase 2).

`build_model` is the full fine-tune / LoRA model builder shared by the VRAM
profiler and the SFT warmup (and, next round, the multi-turn RLOO loop). It is
decoupled from the stale single-turn `shoprl.grpo.trainer` (which imports
un-vendored modules and gets its full multi-turn RLOO rewrite in the next round).
"""
__all__ = ["BuiltModel", "build_model"]


def __getattr__(name):
    # Lazy (PEP 562): build.py imports torch/transformers at module top, but
    # the pure SFT masking helpers (train.sft) must stay importable in the
    # CPU-only test env. GPU callers see no difference.
    if name in __all__:
        from shoprl.train import build
        return getattr(build, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
