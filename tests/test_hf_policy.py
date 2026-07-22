"""HFPolicyV2 interface-level tests — model-free via a stubbed `torch` module
and duck-typed model/tokenizer (the policy's contract is message bookkeeping
around one generate call; the real model is exercised by GPU runs)."""
import contextlib
import sys
import types

import pytest


@pytest.fixture()
def fake_torch(monkeypatch):
    mod = types.ModuleType("torch")
    mod.no_grad = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod


class _Ids:
    def __init__(self, n):
        self.shape = (1, n)

    def to(self, device):
        return self


class _Out:
    def __getitem__(self, key):        # out[0, n:] -> generated token ids
        return [7, 8, 9]


class _Model:
    def __init__(self):
        self.calls = []

    def generate(self, ids, **kw):
        self.calls.append(kw)
        return _Out()


class _Tok:
    pad_token_id = 0

    def __init__(self):
        self.seen_messages = []

    def apply_chat_template(self, messages, **kw):
        self.seen_messages.append([dict(m) for m in messages])
        return {"input_ids": _Ids(len(messages) * 3)}

    def decode(self, ids, **kw):
        return ' {"action": "search", "query": "x"} '


def test_reset_seeds_system_prompt(fake_torch):
    from shoprl.eval.hf_policy import HFPolicyV2
    p = HFPolicyV2(_Model(), _Tok(), system="SYS", device="cpu")
    p.reset()
    assert p.messages == [{"role": "system", "content": "SYS"}]


def test_act_bookkeeping_and_greedy_decode(fake_torch):
    from shoprl.eval.hf_policy import HFPolicyV2
    model, tok = _Model(), _Tok()
    p = HFPolicyV2(model, tok, system="SYS", device="cpu", max_new_tokens=64)
    p.reset()
    text = p.act("hola, necesito una laptop")
    assert text == '{"action": "search", "query": "x"}'   # stripped decode
    # conversation grew: system + user + assistant, in order
    assert [m["role"] for m in p.messages] == ["system", "user", "assistant"]
    assert p.messages[1]["content"] == "hola, necesito una laptop"
    assert p.messages[2]["content"] == text
    # generation contract: greedy, bounded, padded
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["max_new_tokens"] == 64
    # the tokenizer saw the FULL history including the new user turn
    assert tok.seen_messages[0][-1]["role"] == "user"


def test_multi_turn_history_accumulates(fake_torch):
    from shoprl.eval.hf_policy import HFPolicyV2
    p = HFPolicyV2(_Model(), _Tok(), device="cpu")
    p.reset()
    p.act("turn one")
    p.act("turn two")
    assert [m["role"] for m in p.messages] == [
        "system", "user", "assistant", "user", "assistant"]
