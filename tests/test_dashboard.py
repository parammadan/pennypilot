"""Tests for the observability dashboard: data-source abstraction, model build,
and the alert rules (fire on bad data, green on healthy data)."""
from __future__ import annotations

import json

import pytest

from shoprl.observability.datasource import (AWSMetricsSource, LiveTailSource,
                                             StaticFileSource)
from shoprl.observability.dashboard import build_model, render_html


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def _rloo(n, kl_fn, rwd_fn):
    return [{"step": i, "kl_mean": kl_fn(i), "reward_mean": rwd_fn(i),
             "reward_std": 0.05, "grad_norm": 2.0, "ask_rate": 1.0,
             "violation_rate": 0.0} for i in range(n)]


def test_static_and_livetail_read_jsonl(tmp_path):
    path = _write(tmp_path, "m.jsonl", [{"step": 0, "kl_mean": 0.1}])
    for Src in (StaticFileSource, LiveTailSource):
        s = Src(path)
        assert s.available() and len(s.read()) == 1
    assert StaticFileSource(str(tmp_path / "missing.jsonl")).read() == []


def test_aws_source_is_prewired_but_unavailable():
    s = AWSMetricsSource()
    assert s.available() is False and s.read() == []


def test_healthy_run_all_alerts_good(tmp_path):
    rloo = _write(tmp_path, "r.jsonl", _rloo(50, lambda i: 0.00007 * i, lambda i: 0.9 + 0.005 * i))
    sft = _write(tmp_path, "s.jsonl", [{"step": 1, "loss": 3.0}, {"step": 2, "loss": 0.1}])
    m = build_model(StaticFileSource(rloo), StaticFileSource(sft), AWSMetricsSource(),
                    kl_critical=0.5, reward_target=0.8)
    levels = {a["rule"]: a["level"] for a in m["alerts"]}
    assert levels["KL blowup"] == "good"
    assert levels["Reward stall"] == "good"
    assert [c["id"] for c in m["charts"]] == ["kl", "reward", "gradnorm", "loss"]
    # system health = live-ready placeholders (AWS source unavailable)
    assert m["system"]["live"] is False
    assert all(t["value"] is None for t in m["system"]["tiles"])


def test_kl_blowup_is_critical(tmp_path):
    rloo = _write(tmp_path, "r.jsonl", _rloo(30, lambda i: 0.001 * i * i, lambda i: 0.9))
    sft = _write(tmp_path, "s.jsonl", [{"step": 1, "loss": 3.0}])
    m = build_model(StaticFileSource(rloo), StaticFileSource(sft), kl_critical=0.5)
    kl = next(a for a in m["alerts"] if a["rule"] == "KL blowup")
    assert kl["level"] == "critical"


def test_reward_stuck_low_is_warning(tmp_path):
    rloo = _write(tmp_path, "r.jsonl", _rloo(30, lambda i: 0.001, lambda i: 0.2))  # flat + low
    sft = _write(tmp_path, "s.jsonl", [{"step": 1, "loss": 3.0}])
    m = build_model(StaticFileSource(rloo), StaticFileSource(sft), reward_target=0.8)
    rw = next(a for a in m["alerts"] if a["rule"] == "Reward stall")
    assert rw["level"] == "warning"


def test_render_html_embeds_model(tmp_path):
    rloo = _write(tmp_path, "r.jsonl", _rloo(10, lambda i: 0.001, lambda i: 1.0))
    sft = _write(tmp_path, "s.jsonl", [{"step": 1, "loss": 1.0}])
    html = render_html(build_model(StaticFileSource(rloo), StaticFileSource(sft)))
    assert "<!doctype html>" in html and "application/json" in html
    assert '"charts"' in html and "System health" in html
