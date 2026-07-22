"""RemotePolicyV2 against a REAL in-process HTTP server (stdlib only) —
interface-level: message bookkeeping, statelessness of the server contract,
health check. No model, no network beyond localhost."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from shoprl.eval.remote_policy import RemotePolicyV2

RECEIVED: list[dict] = []


class _Stub(BaseHTTPRequestHandler):
    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"ok": True, "ckpt": "stub"})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        RECEIVED.append(payload)
        n = sum(m["role"] == "user" for m in payload["messages"])
        self._send({"text": f'{{"action": "ask_user", "question": "q{n}?"}}'})

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def server_url():
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_health(server_url):
    assert RemotePolicyV2(server_url).health() == {"ok": True, "ckpt": "stub"}


def test_act_appends_and_posts_full_history(server_url):
    RECEIVED.clear()
    p = RemotePolicyV2(server_url, system="SYS")
    p.reset()
    a1 = p.act("hola")
    a2 = p.act("mi presupuesto es $100")
    assert a1 == '{"action": "ask_user", "question": "q1?"}'
    assert a2 == '{"action": "ask_user", "question": "q2?"}'
    # Client keeps the whole conversation; server stays stateless.
    assert [m["role"] for m in RECEIVED[1]["messages"]] == [
        "system", "user", "assistant", "user"]
    assert RECEIVED[1]["messages"][0]["content"] == "SYS"
    assert p.messages[-1]["content"] == a2   # assistant reply recorded client-side


def test_reset_clears_history(server_url):
    p = RemotePolicyV2(server_url)
    p.reset()
    p.act("x")
    p.reset()
    assert [m["role"] for m in p.messages] == ["system"]
