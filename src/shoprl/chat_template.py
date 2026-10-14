"""Chat-template fixes for non-Qwen models (second-model arms, paper F).

Mistral-7B-Instruct-v0.3's official template attaches the system prompt to
the LAST user turn. In a multi-turn episode the system text would then move
every turn, so a conversation prefix is no longer a token prefix of the full
conversation -- which breaks the SFT loss mask (verify_mask) and RL's
assistant-token mask. We pin it to the FIRST user turn instead. Everything
else is byte-identical to the official v0.3 format, and for single-turn
prompts (retention, lm-eval) the output is exactly the official template's.
"""
from __future__ import annotations

MISTRAL_FIRST_SYSTEM = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{%- set system_message = messages[0]['content'] %}"
    "{%- set loop_messages = messages[1:] %}"
    "{%- else %}{%- set loop_messages = messages %}{%- endif %}"
    "{{- bos_token }}"
    "{%- for message in loop_messages %}"
    "{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{- raise_exception('Conversation roles must alternate user/assistant/...') }}"
    "{%- endif %}"
    "{%- if message['role'] == 'user' %}"
    "{%- if loop.first and system_message is defined %}"
    "{{- '[INST] ' + system_message + '\\n\\n' + message['content'] + '[/INST]' }}"
    "{%- else %}{{- '[INST] ' + message['content'] + '[/INST]' }}{%- endif %}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- ' ' + message['content']|trim + eos_token }}"
    "{%- endif %}"
    "{%- endfor %}"
)


def prepare_tokenizer(tok):
    """Idempotent: fix the Mistral template and give it a pad token."""
    t = getattr(tok, "chat_template", None) or ""
    if "[INST]" in t and "loop.last and system_message" in t:
        tok.chat_template = MISTRAL_FIRST_SYSTEM
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    return tok
