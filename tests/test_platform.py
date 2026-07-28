"""PennyData platform: ingestion, derived views, quality, export, API."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from shoprl.platform.export import build_dataset
from shoprl.platform.quality import stats, tag_turn
from shoprl.platform.store import PlatformStore

AJ = '{"action": "ask_user", "question": "What is your budget?"}'


def _episode(store, sid="s1", label="7b", violation=False, feedback=None):
    store.ingest({"kind": "episode_start", "session_id": sid, "label": label,
                  "policy": {"ckpt": "/x/policy"}, "brief": "budget $2000"})
    store.ingest({"kind": "turn", "session_id": sid, "i": 0,
                  "agent": f"Hi there! {AJ}", "observation": "My budget is $2000.",
                  "note": "revealed budget"})
    store.ingest({"kind": "turn", "session_id": sid, "i": 1,
                  "agent": "prose with no action", "observation": "ok",
                  "note": "invalid action (no JSON action object found)"})
    if feedback:
        store.ingest({"kind": "feedback", "session_id": sid, "i": 0,
                      "vote": feedback})
    store.ingest({"kind": "episode_end", "session_id": sid, "verdict": "done",
                  "violation": violation, "cart": ["LAP-0001"]})


def test_ingest_and_derived_views(tmp_path):
    store = PlatformStore(tmp_path)
    _episode(store, feedback="up")
    ep = store.query("SELECT label, violation, cart FROM episodes")
    assert ep["rows"] == [["7b", 0, '["LAP-0001"]']]
    t = store.query("SELECT i, action_kind, feedback FROM turns ORDER BY i")
    assert t["rows"] == [[0, "ask_user", "up"], [1, "invalid", None]]
    # source of truth: nuking the DB and replaying the JSONL reproduces it
    n = store.rebuild_from_log()
    assert n == 5
    assert store.query("SELECT feedback FROM turns WHERE i=0")["rows"] == [["up"]]


def test_feedback_survives_turn_resend(tmp_path):
    store = PlatformStore(tmp_path)
    _episode(store, feedback="down")
    store.ingest({"kind": "turn", "session_id": "s1", "i": 0,
                  "agent": f"Hi there! {AJ}", "observation": "resent", "note": ""})
    assert store.query("SELECT feedback FROM turns WHERE i=0")["rows"] == [["down"]]


def test_query_is_select_only(tmp_path):
    store = PlatformStore(tmp_path)
    with pytest.raises(ValueError):
        store.query("DELETE FROM episodes")
    with pytest.raises(ValueError):
        store.query("SELECT 1; DROP TABLE episodes")


def test_quality_stats_and_tags(tmp_path):
    store = PlatformStore(tmp_path)
    _episode(store, sid="a", label="1.5b", feedback="down")
    _episode(store, sid="b", label="7b", feedback="up")
    st = stats(store)
    assert st["episodes_total"] == 2
    assert st["per_label"]["7b"]["thumbs_up"] == 1
    assert st["per_label"]["1.5b"]["invalid_action_rate"] == 0.5
    assert st["tags"]["invalid_action"] == 2
    assert tag_turn("We only stock laptops here — what's your budget?", "", "") \
        == ["cannot_fulfill_redirect"]
    assert "price_floor_request" in tag_turn(
        "x", "words\n[store notice: price minimum $2500 requested]", "")
    assert "ungrounded_sku" in tag_turn("x", "No such product — pick a SKU", "")


def test_export_excludes_violations_and_reports(tmp_path):
    store = PlatformStore(tmp_path)
    _episode(store, sid="good", feedback="down")
    _episode(store, sid="bad", violation=True)
    out = build_dataset(store)
    assert out["report"]["sequences"] == 1
    assert out["data"][0]["session_id"] == "good"
    assert out["data"][0]["messages"][0]["role"] == "system"
    assert out["report"]["hard_example_candidates"] == 1
    both = build_dataset(store, exclude_violations=False)
    assert both["report"]["sequences"] == 2


def test_http_api_roundtrip(tmp_path):
    from http.server import HTTPServer

    from shoprl.platform.ingest import make_handler
    store = PlatformStore(tmp_path)
    httpd = HTTPServer(("127.0.0.1", 0), make_handler(store))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{port}"
        ev = {"kind": "episode_start", "session_id": "h1", "label": "api"}
        req = urllib.request.Request(f"{base}/events",
                                     data=json.dumps(ev).encode(),
                                     headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req).read())["ok"]
        assert json.loads(urllib.request.urlopen(
            f"{base}/stats").read())["episodes_total"] == 1
        feed = json.loads(urllib.request.urlopen(f"{base}/feed").read())
        assert feed[0]["kind"] == "episode_start"
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "PennyData" in page
        try:
            urllib.request.urlopen(f"{base}/query?sql=DELETE%20FROM%20episodes")
            raise AssertionError("non-SELECT must be rejected")
        except urllib.error.HTTPError as e:
            assert e.code == 400 and "error" in json.loads(e.read())
    finally:
        httpd.shutdown()


def test_ui_events_and_funnel(tmp_path):
    store = PlatformStore(tmp_path)
    _episode(store, sid="f1", label="x")
    store.ingest({"kind": "ui", "session_id": "f1", "type": "hover",
                  "target": "card:LAP-0001"})
    store.ingest({"kind": "ui", "session_id": "f1", "type": "modal",
                  "target": "approve"})
    from shoprl.platform.quality import funnel
    st = stats(store)
    assert st["ui_events"] == {"hover": 1, "modal": 1}
    f = funnel(store)
    assert f["episodes"] == 1 and f["engaged"] == 1
    # rebuild keeps the clickstream (ui_events cleared + replayed)
    store.rebuild_from_log()
    assert stats(store)["ui_events"] == {"hover": 1, "modal": 1}


def test_synth_traffic_direct_mode(tmp_path):
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "scripts/synth_traffic.py", "--n", "40",
         "--root", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    store = PlatformStore(tmp_path)
    st = stats(store)
    assert st["episodes_total"] == 40
    assert st["funnel"]["engaged"] > 0
    assert all(v["violations"] == 0 for v in st["per_label"].values())


def test_s3_archiver_batching_and_layout():
    from shoprl.platform.s3_sink import S3Archiver

    class FakeS3:
        def __init__(self):
            self.puts = []
        def put_object(self, Bucket, Key, Body):
            self.puts.append((Bucket, Key, Body))

    s3 = FakeS3()
    arch = S3Archiver("b", batch_size=3, max_age_s=9999, s3=s3)
    flushed = [arch.add({"kind": "ui", "i": i}) for i in range(7)]
    assert flushed == [False, False, True, False, False, True, False]
    arch.flush()
    assert arch.parts_written == 3 and arch.events_written == 7
    assert all(k.startswith("pennydata/events/date=") and k.endswith(".jsonl")
               for _, k, _ in s3.puts)
    # parts are newline-delimited JSON, replayable
    import json as j
    total = sum(len([l for l in body.decode().splitlines() if l])
                for _, _, body in s3.puts)
    assert total == 7 and j.loads(s3.puts[0][2].decode().splitlines()[0])["i"] == 0
