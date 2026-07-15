"""Training-side building blocks for Pennywise (Phase 2).

`build_model` is the full fine-tune / LoRA model builder shared by the VRAM
profiler and the SFT warmup (and, next round, the multi-turn RLOO loop). It is
decoupled from the stale single-turn `shoprl.grpo.trainer` (which imports
un-vendored modules and gets its full multi-turn RLOO rewrite in the next round).
"""
from shoprl.train.build import BuiltModel, build_model

__all__ = ["BuiltModel", "build_model"]
