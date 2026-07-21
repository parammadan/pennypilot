"""Capture harness: sampler/events/phases/figures/manifest + the
prediction-before-measurement rule — all CPU (fake sample_fn, Agg backend)."""
import time

import pytest

from shoprl.profiling import (EventLog, GPUSampler, phase_breakdown,
                              render_phase_bar, render_timeline,
                              require_prediction, write_manifest)


def test_sampler_with_fake_fn_collects_and_writes(tmp_path):
    s = GPUSampler(interval=0.01, sample_fn=lambda: (55.0, 7.3)).start()
    time.sleep(0.08)
    samples = s.stop()
    assert len(samples) >= 3
    assert all(u == 55.0 and m == 7.3 for _, u, m in samples)
    s.to_csv(tmp_path / "gpu.csv")
    assert (tmp_path / "gpu.csv").read_text().startswith("t_s,gpu_util_pct,mem_gb")


def test_phase_breakdown_pairs_marks():
    events = [(0.0, "rollout:start"), (8.0, "rollout:end"),
              (8.0, "update:start"), (9.5, "update:end"),
              (9.5, "optimizer:start"), (10.0, "optimizer:end")]
    b = phase_breakdown(events)
    assert b["phases"] == {"rollout": 8.0, "update": 1.5, "optimizer": 0.5}
    assert b["fractions"]["rollout"] == pytest.approx(0.8)
    assert b["window_s"] == 10.0


def test_figures_render_png_and_csv_twins(tmp_path):
    samples = [(i * 0.2, 40 + (i % 5) * 10, 5 + 0.02 * i) for i in range(60)]
    events = [(1.0, "rollout:start"), (9.0, "rollout:end")]
    png = tmp_path / "ss00_timeline.png"
    render_timeline(samples, events, png,
                    "SS0 — baseline | predicted: rollout ≥70% | measured: 80%")
    assert png.exists() and png.stat().st_size > 10_000
    assert (tmp_path / "ss00_timeline.csv").exists()
    bar = tmp_path / "ss00_phases.png"
    render_phase_bar(phase_breakdown(events + [(9.0, "update:start"),
                                               (10.0, "update:end")]),
                     bar, "SS0 — phase split")
    assert bar.exists() and (tmp_path / "ss00_phases.json").exists()


def test_prediction_is_mandatory():
    with pytest.raises(SystemExit):
        require_prediction("")
    with pytest.raises(SystemExit):
        require_prediction("   ")
    assert require_prediction(" rollout dominates ") == "rollout dominates"


def test_manifest_contents(tmp_path):
    (tmp_path / "x.png").write_bytes(b"png")
    p = write_manifest(tmp_path, "SS0", "baseline iteration split",
                       predicted="rollout >= 70%", measured="rollout 81%",
                       mechanism="autoregressive decode is bandwidth-bound",
                       config={"k": 8}, rerun_cmd="python benchmarks/ss00.py")
    text = p.read_text()
    for needle in ("SS0", "rollout >= 70%", "rollout 81%", "hash", "x.png",
                   "Rerun"):
        assert needle in text
