"""Self-service platform: auth, extraction jobs, training submission, logs,
privacy scrubbing, SQL governance."""
import json
import urllib.error
import urllib.request

import pytest

from shoprl.platform.jobs import JobService
from shoprl.platform.privacy import scrub
from shoprl.platform.store import PlatformStore


def test_privacy_scrub_types_and_counts():
    text = ("email me at a.b@x.com or call 555-123-4567, card "
            "4539 1488 0343 6467, ssn 123-45-6789, at 42 Main St")
    clean, counts = scrub(text)
    assert "<EMAIL>" in clean and "<PHONE>" in clean and "<CARD>" in clean
    assert "<SSN>" in clean and "<ADDRESS>" in clean
    assert counts == {"EMAIL": 1, "PHONE": 1, "SSN": 1, "ADDRESS": 1, "CARD": 1}
    # non-Luhn digit runs are NOT cards
    c2, n2 = scrub("order number 1234 5678 9012 3456 7")
    assert "CARD" not in n2


def test_ingest_scrubs_customer_text(tmp_path):
    store = PlatformStore(tmp_path)
    store.ingest({"kind": "episode_start", "session_id": "p1", "label": "m"})
    store.ingest({"kind": "turn", "session_id": "p1", "i": 0, "agent": "hi",
                  "observation": "my email is leak@real.com ok?"})
    obs = store.query("SELECT observation FROM turns")["rows"][0][0]
    assert "leak@real.com" not in obs and "<EMAIL>" in obs
    n = store.query("SELECT SUM(n) FROM privacy_log")["rows"][0][0]
    assert n == 1


def test_jobs_auth_extraction_training(tmp_path, monkeypatch):
    monkeypatch.chdir("/Users/parammadan/pennywise-v100-infra")
    store = PlatformStore(tmp_path / "s")
    svc = JobService(store, dataset_dir=tmp_path / "ds",
                     slurm_submit=lambda p: "999001",
                     slurm_status=lambda j: "COMPLETED")
    assert svc.login("param", "wrong") is None
    tok = svc.login("param", "pennydata")
    assert tok and svc.whoami(tok["token"])["role"] == "admin"
    # extraction against the committed approved recipe (empty store -> 0 seqs)
    job = svc.request_extraction("param", recipe_id="premature-search-v3")
    assert job["status"] == "succeeded"
    assert job["result"]["manifest"]["sequences"] == 0
    assert job["result"]["manifest"]["extraction_source"] == "hot"
    # training submission with injected slurm
    tj = svc.request_training("param", dataset=str(
        tmp_path / "ds" / "premature-search-v3.jsonl"), out_name="ui_test")
    assert tj["status"] == "submitted"
    assert tj["result"]["slurm_id"] == "999001"
    assert {j["kind"] for j in svc.jobs()} == {"extraction", "training"}
    assert all(j["requested_by"] == "param" for j in svc.jobs())


def test_sql_is_admin_gated(tmp_path, monkeypatch):
    import threading
    from http.server import HTTPServer
    from shoprl.platform.ingest import make_handler
    monkeypatch.chdir("/Users/parammadan/pennywise-v100-infra")
    store = PlatformStore(tmp_path)
    httpd = HTTPServer(("127.0.0.1", 0), make_handler(store))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"{base}/query?sql=SELECT%201")
        assert e.value.code == 403                    # no login, no SQL
        req = urllib.request.Request(
            f"{base}/login", data=json.dumps(
                {"user": "param", "password": "pennydata"}).encode(),
            headers={"Content-Type": "application/json"})
        tok = json.loads(urllib.request.urlopen(req).read())["token"]
        req = urllib.request.Request(f"{base}/query?sql=SELECT%201",
                                     headers={"Authorization": f"Bearer {tok}"})
        assert json.loads(urllib.request.urlopen(req).read())["rows"] == [[1]]
        logs = json.loads(urllib.request.urlopen(f"{base}/logs").read())
        assert logs["logins"][0]["user"] == "param"
    finally:
        httpd.shutdown()
