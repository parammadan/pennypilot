"""bench_common interface tests — GenMeter arithmetic and the two agent_fn
factories' bookkeeping (stubbed torch/vllm; real engines run on GPU)."""
import contextlib
import sys
import types

import pytest

from shoprl.profiling.bench_common import GenMeter


def test_genmeter_math():
    m = GenMeter()
    assert m.engine_tps() == 0.0            # no div-by-zero on empty
    m.gen_tokens, m.gen_seconds, m.calls = 500, 2.0, 3
    assert m.engine_tps() == 250.0


def test_hf_agent_fn_meters_and_decodes(monkeypatch):
    class _NoGrad:                       # usable as decorator AND context mgr
        def __call__(self, fn):
            return fn

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = _NoGrad
    fake_torch.cuda = types.SimpleNamespace(synchronize=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    from shoprl.profiling.bench_common import hf_agent_fn

    class Ids:
        shape = (1, 4)

        def to(self, d):
            return self

    class Out:
        shape = (1, 10)                     # 6 generated tokens

        def __getitem__(self, k):
            return [1] * 6

    tok = types.SimpleNamespace(
        pad_token_id=0,
        apply_chat_template=lambda msgs, **kw: {"input_ids": Ids()},
        decode=lambda ids, **kw: "  act  ")
    model = types.SimpleNamespace(generate=lambda ids, **kw: Out())
    meter = GenMeter()
    fn = hf_agent_fn(model, tok, meter)
    assert fn([{"role": "user", "content": "x"}]) == "act"
    assert meter.gen_tokens == 6 and meter.calls == 1
    assert meter.gen_seconds > 0


def test_vllm_agent_fn_meters_and_prompts(monkeypatch):
    fake_vllm = types.ModuleType("vllm")

    class SP:
        def __init__(self, **kw):
            self.kw = kw
    fake_vllm.SamplingParams = SP
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    from shoprl.profiling.bench_common import vllm_agent_fn

    prompts_seen = []
    out = types.SimpleNamespace(outputs=[types.SimpleNamespace(
        token_ids=[1, 2, 3], text=" go ")])
    llm = types.SimpleNamespace(
        generate=lambda ps, sp, **kw: (prompts_seen.extend(ps), [out])[1])
    tok = types.SimpleNamespace(
        apply_chat_template=lambda msgs, **kw: f"PROMPT[{len(msgs)}]")
    meter = GenMeter()
    fn = vllm_agent_fn(llm, tok, meter)     # no adapter -> no LoRARequest import
    assert fn([{"role": "user", "content": "x"}]) == "go"
    assert prompts_seen == ["PROMPT[1]"]
    assert meter.gen_tokens == 3 and meter.calls == 1
