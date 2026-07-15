"""SFT warmup trainer: teach the ritual before RL.

Renders each `Demo` (from `data/sft.py`) into the model's chat template and
trains with the loss **masked to the agent (assistant) turns only** — the model
learns to PRODUCE the agent's actions, not to model the user's utterances. This
is supervised warmup, not RL: it gives the policy the ask→discover→recommend→
ask-permission→add ritual so RL has something to sharpen.

Masking uses the prefix-difference method (works with any chat template): for
each assistant message, its token span is `template(msgs[:i+1]) minus
template(msgs[:i] + generation_prompt)`, and only that span is unmasked.
"""
from __future__ import annotations

import torch

from shoprl.data.sft import Demo

_ROLE = {"agent": "assistant", "user": "user"}


def demo_to_messages(demo: Demo) -> list[dict]:
    return [{"role": _ROLE[t.role], "content": t.text} for t in demo.turns]


def encode(tokenizer, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    """Chat-template -> flat list of token ids. transformers 5.13 returns a
    BatchEncoding (a UserDict, NOT a dict subclass), so we force return_dict and
    read input_ids explicitly rather than relying on isinstance(_, dict)."""
    enc = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=True,
        add_generation_prompt=add_generation_prompt)
    ids = enc["input_ids"]
    if ids and isinstance(ids[0], list):  # accidental batch nesting
        ids = ids[0]
    return list(ids)


def build_example(tokenizer, demo: Demo, max_len: int) -> dict:
    """Return {input_ids, labels} with labels = -100 except on agent spans."""
    messages = demo_to_messages(demo)
    full = encode(tokenizer, messages, add_generation_prompt=False)
    labels = [-100] * len(full)
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix = encode(tokenizer, messages[:i], add_generation_prompt=True)
        upto = encode(tokenizer, messages[:i + 1], add_generation_prompt=False)
        for j in range(len(prefix), min(len(upto), len(full))):
            labels[j] = full[j]
    return {"input_ids": full[:max_len], "labels": labels[:max_len]}


def collate(batch: list[dict], pad_id: int, device: str) -> dict:
    width = max(len(ex["input_ids"]) for ex in batch)
    ids, labs, attn = [], [], []
    for ex in batch:
        n = len(ex["input_ids"])
        pad = width - n
        ids.append(ex["input_ids"] + [pad_id] * pad)
        labs.append(ex["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
    t = lambda x: torch.tensor(x, device=device)
    return {"input_ids": t(ids), "attention_mask": t(attn), "labels": t(labs)}


def train_sft(built, demos: list[Demo], *, max_len: int, batch_size: int = 4,
              epochs: int = 1, max_steps: int | None = None, lr: float = 1e-5,
              log_every: int = 10):
    """Bounded, observed SFT. Yields a metrics dict per step (so the caller can
    log/print live). `built` is a BuiltModel; `built.policy` is trained."""
    model, tok, device = built.policy, built.tokenizer, built.device
    examples = [build_example(tok, d, max_len) for d in demos]
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr)
    model.train()

    step = 0
    for _ in range(epochs):
        for i in range(0, len(examples), batch_size):
            batch = collate(examples[i:i + batch_size], tok.pad_token_id, device)
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            supervised = int((batch["labels"] != -100).sum().item())
            yield {"step": step, "loss": float(out.loss.item()),
                   "supervised_tokens": supervised,
                   "batch_tokens": int(batch["attention_mask"].sum().item())}
            if max_steps and step >= max_steps:
                return
