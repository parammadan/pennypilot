"""Prove the 7B chat-face does a full chat -> clarify -> search -> recommend
-> permission -> cart flow through the real RemotePolicyV2 client.

    python scripts/probe_chat_flow.py http://<node>:8765
"""
import sys

from shoprl.data.catalog import generate_catalog
from shoprl.data.prompts_v2 import SYSTEM_PROMPT_CHAT
from shoprl.eval.remote_policy import RemotePolicyV2

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
cat = generate_catalog(n=150, seed=0)
top = sorted(cat, key=lambda p: p.price)[:4]
results = "Matching products (cheapest first):\n" + "\n".join(
    f"- {p.sku}: ${p.price:.0f}, {p.ram_gb}GB RAM, {p.weight_lbs}lbs, "
    f"{p.battery_hrs}hrs, {p.brand}" for p in top)

p = RemotePolicyV2(URL, system=SYSTEM_PROMPT_CHAT)
p.reset()
script = [
    ("You", "hi! I need a laptop"),
    ("You", "my budget is about 1200 dollars"),
    ("You", "16GB RAM, lightweight, any brand is fine"),
    ("STORE (search results fed back)", results),
    ("You", "yes please, add it"),
]
for tag, msg in script:
    shown = msg if len(msg) <= 70 else msg[:70] + "..."
    print(f"\n[{tag}] {shown}")
    print(f"PennyPilot: {p.act(msg)}")
