"""End-to-end vertical slice on CPU: campaign (fake agents) → ingestion →
milestone funnel → friction metrics → attribution — the full evidence chain
the GPU run will produce with real checkpoints."""
import json
import subprocess
import sys

from shoprl.platform.attribution import attribute_all
from shoprl.platform.milestones import friction_metrics, milestone_funnel
from shoprl.platform.store import PlatformStore


def _run(tmp_path, fake, label, n=12):
    out = tmp_path / f"{label}.jsonl"
    r = subprocess.run(
        [sys.executable, "scripts/behavior_campaign.py", "--fake-policy", fake,
         "--label", label, "--n", str(n), "--seed", "100",
         "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    return out


def test_full_slice_two_agents(tmp_path):
    good = _run(tmp_path, "good", "agent-good")
    rude = _run(tmp_path, "rude", "agent-rude")
    store = PlatformStore(tmp_path / "store")
    for f in (good, rude):
        for line in open(f):
            store.ingest(json.loads(line))
    # idempotency across the whole file: re-ingest changes nothing
    before = store.query("SELECT COUNT(*) FROM events")["rows"][0][0]
    for line in open(rude):
        assert store.ingest(json.loads(line)) == "duplicate"
    assert store.query("SELECT COUNT(*) FROM events")["rows"][0][0] == before

    f = milestone_funnel(store)
    bym = f["by_model_version"]
    good_f = {s["stage"]: s["reached"] for s in bym["agent-good"]["stages"]}
    rude_f = {s["stage"]: s["reached"] for s in bym["agent-rude"]["stages"]}
    assert good_f["task_satisfied"] == 12          # good agent completes all
    assert rude_f["goal_understood"] < 12          # rude agent fails early
    assert rude_f["task_satisfied"] == 0
    reasons = bym["agent-rude"]["failure_reasons"]["goal_understood"]
    assert reasons, "reason codes present for the failing stage"

    m = friction_metrics(store)
    p = m["premature_search_rate"]["by_model_version"]
    assert p["agent-rude"]["value"] == 1.0
    assert p["agent-good"]["value"] == 0.0
    c = m["customer_correction_rate"]["by_model_version"]
    assert c["agent-rude"]["value"] > 0.5
    assert c["agent-good"]["value"] == 0.0

    atts = attribute_all(store)                    # failed sessions only
    fired = [a for a in atts
             if a["primary_category"] == "CONSTRAINT_EXTRACTION"]
    assert fired, "attribution fires on rude-agent failures"
    assert all(a["evidence_event_ids"] for a in fired)
    assert all(a["session_id"].startswith("camp-agent-rude")
               for a in fired)
    # validation against the simulator's hidden causes
    simlog = [json.loads(l) for l in
              open(str(rude).replace(".jsonl", "_simlog.jsonl"))]
    reasons = {r["reason"] for s in simlog for r in s["log"]}
    assert "AGENT_SEARCHED_PREMATURELY" in reasons
