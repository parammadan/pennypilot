"""Model builder: full fine-tune (target) or LoRA (measured fallback).

Volta (V100, cap 7.0) has no bf16 tensor cores, so fp16 is the default compute
dtype. The key structural difference between the two methods is the KL reference:

  - LoRA: the frozen base doubles as the reference for free (disable the adapter
    to get reference log-probs), so no second model is needed.
  - Full fine-tune: every weight moves, so the base is no longer a valid frozen
    reference. We load a SEPARATE frozen copy of the base as the KL reference.
    (1.5B in fp16 ≈ 3 GB — affordable on a 32 GB card, but it is part of the
    memory budget the profiler measures.)

This builder is deliberately decoupled from the stale single-turn
`shoprl.grpo.trainer`; the profiler and SFT warmup use it directly, and the
multi-turn RLOO loop will too (next round).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BuiltModel:
    policy: object
    reference: object | None   # frozen KL reference (full-FT); None for LoRA
    tokenizer: object
    device: str
    method: str

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.policy.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.policy.parameters())


def build_model(
    model_name: str,
    method: str = "full",
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    grad_checkpointing: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> BuiltModel:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)

    if method == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
            target_modules="all-linear", task_type="CAUSAL_LM",
        )
        policy = get_peft_model(base, cfg).to(device)
        reference = None  # disable_adapter() gives the reference for free
        if grad_checkpointing:
            policy.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            policy.enable_input_require_grads()
        policy.config.use_cache = False

    elif method == "full":
        policy = base.to(device)
        for p in policy.parameters():
            p.requires_grad_(True)
        policy.config.use_cache = False
        if grad_checkpointing:
            policy.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        # Separate frozen KL reference (a moved full-FT policy can't be its own).
        reference = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)

    else:
        raise ValueError(f"unknown method {method!r} (expected 'full' or 'lora')")

    return BuiltModel(policy=policy, reference=reference, tokenizer=tokenizer,
                      device=device, method=method)
