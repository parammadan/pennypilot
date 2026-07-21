"""Loss-mask correctness (spec §8) — the silent-failure mode that must crash.

Uses a deterministic fake chat-template tokenizer so the prefix-difference
masking in train/sft.py is exercised end-to-end on real v2 demos, model-free.
The GPU path re-runs the same `verify_mask` with the real tokenizer at the
start of every SFT run (train_sft calls it on the first example).
"""
import pytest

from shoprl.data.catalog import generate_catalog
from shoprl.data.sft_v2 import generate_sft_v2_dialogues
from shoprl.train.sft import build_example, verify_mask


class FakeChatTok:
    """Word-level chat-template tokenizer: <|role|> words... <eot> per message,
    trailing <|assistant|> when add_generation_prompt=True. Deterministic and
    reversible, so mask spans can be checked exactly."""
    pad_token = "<pad>"
    pad_token_id = 0

    def __init__(self):
        self._vocab = {"<pad>": 0}
        self._toks = ["<pad>"]

    def _id(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = len(self._toks)
            self._toks.append(tok)
        return self._vocab[tok]

    def apply_chat_template(self, messages, tokenize=True, return_dict=False,
                            add_generation_prompt=False):
        toks: list[str] = []
        for m in messages:
            toks += [f"<|{m['role']}|>"] + m["content"].split() + ["<eot>"]
        if add_generation_prompt:
            toks.append("<|assistant|>")
        return {"input_ids": [self._id(t) for t in toks]}

    def decode(self, ids):
        return " ".join(self._toks[i] for i in ids
                        if self._toks[i] not in ("<pad>",))


@pytest.fixture(scope="module")
def demos():
    catalog = generate_catalog(n=300, seed=0)
    return generate_sft_v2_dialogues(catalog, n=12, seed=0)


def test_mask_covers_exactly_the_agent_turns(demos):
    tok = FakeChatTok()
    for demo in demos:
        verify_mask(tok, demo, max_len=4096)   # raises loudly on any leak


def test_supervised_fraction_is_sane(demos):
    tok = FakeChatTok()
    for demo in demos:
        ex = build_example(tok, demo, max_len=4096)
        sup = sum(l != -100 for l in ex["labels"])
        # agent JSON actions are a minority of tokens (user turns + search
        # results dominate) — a mask covering most of the sequence is wrong.
        assert 0 < sup < 0.6 * len(ex["labels"])


def test_corrupted_mask_fails_loudly(demos):
    tok = FakeChatTok()
    demo = demos[0]
    ex = build_example(tok, demo, max_len=4096)

    # Corruption A: unmask one user/env token -> leak must be caught.
    leaked = dict(ex)
    labels = list(ex["labels"])
    user_span = [j for j, l in enumerate(labels) if l == -100]
    # unmask a contiguous run inside a user turn (long enough to be >12 chars)
    start = user_span[len(user_span) // 2]
    for j in range(start, min(start + 12, len(labels))):
        labels[j] = ex["input_ids"][j]
    leaked["labels"] = labels
    with pytest.raises(AssertionError):
        verify_mask(tok, demo, example=leaked)

    # Corruption B: misaligned labels (shifted by one) -> caught by identity check.
    shifted = dict(ex)
    shifted["labels"] = [-100] + [ex["input_ids"][j - 1] if ex["labels"][j] != -100
                                  else -100 for j in range(1, len(ex["labels"]))]
    if any(l != -100 for l in shifted["labels"]):
        with pytest.raises(AssertionError):
            verify_mask(tok, demo, example=shifted)

    # Corruption C: mask everything -> agent turns missing from the span.
    allmasked = dict(ex)
    allmasked["labels"] = [-100] * len(ex["labels"])
    with pytest.raises(AssertionError):
        verify_mask(tok, demo, example=allmasked)


def test_system_prompt_is_masked_and_verifies(demos):
    from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2
    tok = FakeChatTok()
    demo = demos[0]
    ex = build_example(tok, demo, max_len=8192, system=SYSTEM_PROMPT_V2)
    verify_mask(tok, demo, max_len=8192, example=ex, system=SYSTEM_PROMPT_V2)
    # The system span sits at the front and must be fully masked.
    sys_len = len(tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT_V2}])["input_ids"])
    assert all(l == -100 for l in ex["labels"][:sys_len])
    # And the example is strictly longer than the no-system one.
    assert len(ex["input_ids"]) > len(build_example(tok, demo, 8192)["input_ids"])
