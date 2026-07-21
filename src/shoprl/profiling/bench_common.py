"""Shared pieces for the SS benchmarks (GPU; imports are lazy).

Both rollout engines are exposed as `agent_fn(messages) -> str` so SS1 (HF)
and SS2 (vLLM) drive the IDENTICAL environment loop (`rollout_v2`) — the only
variable is the engine, which is the point of the comparison. Each agent_fn
also tallies generated tokens and pure generation seconds so the benchmarks
can report engine throughput separately from loop-effective throughput.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class GenMeter:
    gen_tokens: int = 0
    gen_seconds: float = 0.0
    calls: int = 0

    def engine_tps(self) -> float:
        return round(self.gen_tokens / self.gen_seconds, 1) if self.gen_seconds else 0.0


def load_hf_policy(model_name: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(adapter or model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, attn_implementation="sdpa").to("cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def hf_agent_fn(model, tok, meter: GenMeter, max_new_tokens: int = 64,
                temperature: float = 1.0):
    import torch

    @torch.no_grad()
    def agent_fn(messages):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      tokenize=True, return_dict=True,
                                      return_tensors="pt")
        ids = enc["input_ids"].to("cuda")
        t0 = time.time()
        out = model.generate(ids, max_new_tokens=max_new_tokens,
                             do_sample=temperature > 0,
                             temperature=max(temperature, 1e-5), top_p=0.95,
                             pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        meter.gen_seconds += time.time() - t0
        meter.gen_tokens += int(out.shape[1] - ids.shape[1])
        meter.calls += 1
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    return agent_fn


def vllm_agent_fn(llm, tok, meter: GenMeter, adapter: str | None = None,
                  max_new_tokens: int = 64, temperature: float = 1.0):
    """vLLM-backed agent_fn (run inside the pinned vllm venv). The adapter
    rides along as a LoRARequest so the POLICY, not the base, generates."""
    from vllm import SamplingParams
    lora_req = None
    if adapter:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("policy", 1, adapter)
    sp = SamplingParams(temperature=temperature, top_p=0.95,
                        max_tokens=max_new_tokens)

    def agent_fn(messages):
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
        t0 = time.time()
        outs = llm.generate([prompt], sp, lora_request=lora_req, use_tqdm=False)
        meter.gen_seconds += time.time() - t0
        meter.gen_tokens += len(outs[0].outputs[0].token_ids)
        meter.calls += 1
        return outs[0].outputs[0].text.strip()

    return agent_fn
