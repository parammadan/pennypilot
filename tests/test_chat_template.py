"""Needs the cached HF tokenizers (cluster); skipped elsewhere."""
import pytest

transformers = pytest.importorskip("transformers")
from shoprl.chat_template import prepare_tokenizer  # noqa: E402

MISTRAL, QWEN = "mistralai/Mistral-7B-Instruct-v0.3", "Qwen/Qwen2.5-7B-Instruct"
SYS = [{"role": "system", "content": "You are PennyPilot."}]
CONV = SYS + [{"role": "user", "content": "hola, need a laptop"},
              {"role": "assistant", "content": 'Sure! {"action": "ask_user", "question": "Budget?"}'},
              {"role": "user", "content": "under 900"},
              {"role": "assistant", "content": 'Ok. {"action": "search", "query": "laptop"}'}]


def _tok(name):
    try:
        return transformers.AutoTokenizer.from_pretrained(name)
    except Exception:
        pytest.skip(f"{name} not cached")


def test_mistral_single_turn_identical_to_official():
    off, ours = _tok(MISTRAL), prepare_tokenizer(_tok(MISTRAL))
    m = SYS + [{"role": "user", "content": "What is the capital of Australia?"}]
    assert ours.apply_chat_template(m, tokenize=False) == off.apply_chat_template(m, tokenize=False)


def test_mistral_multi_turn_prefix_property():
    tok = prepare_tokenizer(_tok(MISTRAL))
    for i, msg in enumerate(CONV):
        if msg["role"] != "assistant":
            continue
        pre = tok.apply_chat_template(CONV[:i], add_generation_prompt=True,
                                      tokenize=True, return_dict=True)["input_ids"]
        full = tok.apply_chat_template(CONV[:i + 1], tokenize=True,
                                       return_dict=True)["input_ids"]
        assert len(full) > len(pre)
        assert full[:len(pre)] == pre
    assert tok.pad_token is not None


def test_qwen_untouched():
    a, b = _tok(QWEN), prepare_tokenizer(_tok(QWEN))
    assert a.chat_template == b.chat_template and a.pad_token == b.pad_token
