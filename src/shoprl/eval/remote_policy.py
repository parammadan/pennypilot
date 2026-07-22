"""Remote policy client — same interface as HFPolicyV2, generation over HTTP.

Talks to scripts/serve_policy.py (usually through an SSH tunnel:
http://localhost:8765). Keeps the conversation client-side and posts the full
message list each turn, so the server stays stateless.
"""
from __future__ import annotations

import json
import urllib.request

from shoprl.data.prompts_v2 import SYSTEM_PROMPT_V2

# Bypass any ambient http(s)_proxy: the policy server is reached over localhost
# (SSH tunnel) or an internal cluster node — routing those through an egress
# proxy returns 503. An empty ProxyHandler forces a direct connection.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class RemotePolicyV2:
    def __init__(self, url: str = "http://localhost:8765",
                 system: str = SYSTEM_PROMPT_V2, timeout: float = 120.0):
        self.url = url.rstrip("/")
        self.system = system
        self.timeout = timeout
        self.messages: list[dict] = []

    def health(self) -> dict:
        with _OPENER.open(f"{self.url}/health", timeout=self.timeout) as r:
            return json.loads(r.read())

    def reset(self, scenario=None, idx=None) -> None:
        self.messages = [{"role": "system", "content": self.system}]

    def act(self, observation: str = "") -> str:
        self.messages.append({"role": "user", "content": observation or ""})
        req = urllib.request.Request(
            f"{self.url}/act",
            data=json.dumps({"messages": self.messages}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with _OPENER.open(req, timeout=self.timeout) as r:
            text = json.loads(r.read())["text"]
        self.messages.append({"role": "assistant", "content": text})
        return text
