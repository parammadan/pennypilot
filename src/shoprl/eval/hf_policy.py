"""HF-backed policy for the v2 eval harness (and, later, RL rollouts).

Scenario-agnostic: it only ever sees the conversation (system prompt + user/env
observations) — never the hidden scenario — so it slots into the same harness
as the scripted oracles and compares like-for-like. torch/transformers imports
are lazy so the module is importable in the CPU test env.
"""
from __future__ import annotations

from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2


class HFPolicyV2:
    """Greedy next-action policy over a (possibly LoRA-adapted) causal LM."""

    def __init__(self, model, tokenizer, system: str = SYSTEM_PROMPT_V2,
                 max_new_tokens: int = 64, device: str = "cuda"):
        self.model = model
        self.tok = tokenizer
        self.system = system
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.messages: list[dict] = []

    def reset(self, scenario=None, idx=None) -> None:
        # Oracle-parity signature; a model policy ignores the hidden scenario.
        self.messages = [{"role": "system", "content": self.system}]

    def act(self, observation: str = "") -> str:
        import torch

        self.messages.append({"role": "user", "content": observation or ""})
        enc = self.tok.apply_chat_template(self.messages,
                                           add_generation_prompt=True,
                                           tokenize=True, return_dict=True,
                                           return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens,
                                      do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        text = self.tok.decode(out[0, ids.shape[1]:],
                               skip_special_tokens=True).strip()
        self.messages.append({"role": "assistant", "content": text})
        return text
