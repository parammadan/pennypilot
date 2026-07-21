"""Stage-6 capture harness — every benchmark saves its own evidence.

One reusable module per the campaign rules (docs repo STAGE6_BENCHMARKS.md):
  - GPUSampler   : 200 ms NVML poller (memory GB, GPU util %) with an
                   injectable sample_fn (CPU tests use a fake; on the cluster
                   pynvml, falling back to `nvidia-smi --query-gpu`).
  - EventLog     : timestamped phase annotations ("rollout:start", …).
  - render_*     : 300-dpi figures + the CSV twin next to each PNG. Titles
                   carry the caption format `SSNN — what | predicted: X |
                   measured: Y`; two measures of different scale get two
                   stacked panels, never a dual axis.
  - require_prediction / write_manifest : the artifact policy — a benchmark
                   REFUSES to run with an empty prediction, and every artifact
                   dir gets a MANIFEST.md (what ran, config, predicted vs
                   measured, mechanism, rerun command).

Chart colors are the validated defaults (light surface): util #2a78d6 (blue),
memory #eb6834 (orange); annotations/ink in text grays.
"""
from __future__ import annotations

import csv
import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

_UTIL_COLOR = "#2a78d6"
_MEM_COLOR = "#eb6834"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_SURFACE = "#fcfcfb"
_GRID = "#e7e7e4"


def _nvml_sample():
    import pynvml
    if not getattr(_nvml_sample, "_init", False):
        pynvml.nvmlInit()
        _nvml_sample._init = True
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    util = pynvml.nvmlDeviceGetUtilizationRates(h)
    return float(util.gpu), mem.used / 2**30


def _smi_sample():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
    util, mem_mb = (float(x.strip()) for x in out.split(","))
    return util, mem_mb / 1024.0


def default_sample_fn():
    try:
        return _nvml_sample()
    except Exception:
        return _smi_sample()


class GPUSampler:
    """Background poller: (t_seconds, util_pct, mem_gb) every `interval` s."""

    def __init__(self, interval: float = 0.2, sample_fn=None):
        self.interval = interval
        self.sample_fn = sample_fn or default_sample_fn
        self.samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._t0 = None
        self._thread: threading.Thread | None = None

    def start(self) -> "GPUSampler":
        self._t0 = time.time()
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                try:
                    util, mem = self.sample_fn()
                    self.samples.append((time.time() - self._t0, util, mem))
                except Exception:
                    pass
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> list[tuple[float, float, float]]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.samples

    @property
    def t0(self) -> float | None:
        return self._t0

    def to_csv(self, path) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "gpu_util_pct", "mem_gb"])
            w.writerows(self.samples)


@dataclass
class EventLog:
    """Phase annotations relative to a t0 (share the sampler's)."""
    t0: float = field(default_factory=time.time)
    events: list[tuple[float, str]] = field(default_factory=list)

    def mark(self, label: str) -> None:
        self.events.append((time.time() - self.t0, label))

    def to_csv(self, path) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "label"])
            w.writerows(self.events)


def phase_breakdown(events: list[tuple[float, str]]) -> dict:
    """Pair `<phase>:start` / `<phase>:end` marks into wall-clock totals +
    fractions of the spanned window — SS0's headline numbers."""
    totals: dict[str, float] = {}
    open_at: dict[str, float] = {}
    for t, label in events:
        if label.endswith(":start"):
            open_at[label[:-6]] = t
        elif label.endswith(":end"):
            phase = label[:-4]
            if phase in open_at:
                totals[phase] = totals.get(phase, 0.0) + (t - open_at.pop(phase))
    if not events:
        return {"phases": {}, "fractions": {}, "window_s": 0.0}
    window = max(t for t, _ in events) - min(t for t, _ in events)
    fractions = {k: (v / window if window else 0.0) for k, v in totals.items()}
    return {"phases": {k: round(v, 2) for k, v in totals.items()},
            "fractions": {k: round(v, 4) for k, v in fractions.items()},
            "window_s": round(window, 2)}


# ---- figures -----------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(_SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_INK_2, labelsize=8)


def render_timeline(samples, events, out_png, title: str,
                    subtitle: str = "") -> None:
    """Two stacked panels (util %, memory GB) over time with phase-annotation
    vlines — one axis per measure, shared time base. Writes the CSV twin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = [s[0] for s in samples]
    util = [s[1] for s in samples]
    mem = [s[2] for s in samples]
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11, 5.6),
                                   facecolor=_SURFACE)
    ax1.plot(ts, util, color=_UTIL_COLOR, linewidth=2)
    ax1.set_ylabel("GPU util (%)", color=_INK, fontsize=9)
    ax1.set_ylim(0, 105)
    ax2.plot(ts, mem, color=_MEM_COLOR, linewidth=2)
    ax2.set_ylabel("Memory (GB)", color=_INK, fontsize=9)
    ax2.set_xlabel("seconds", color=_INK_2, fontsize=9)
    for ax in (ax1, ax2):
        _style(ax)
        for t, label in events:
            ax.axvline(t, color=_INK_2, linewidth=0.8, alpha=0.55,
                       linestyle=(0, (3, 2)))
    for t, label in events:                      # label once, on the top panel
        ax1.annotate(label, (t, 103), rotation=90, fontsize=6.5,
                     color=_INK_2, va="top", ha="right", clip_on=False)
    fig.suptitle(title, fontsize=11, color=_INK, x=0.01, ha="left")
    if subtitle:
        ax1.set_title(subtitle, fontsize=8.5, color=_INK_2, loc="left", pad=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=300, facecolor=_SURFACE)
    plt.close(fig)
    csv_path = str(Path(out_png).with_suffix(".csv"))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "gpu_util_pct", "mem_gb"])
        w.writerows(samples)


def render_phase_bar(breakdown: dict, out_png, title: str,
                     subtitle: str = "") -> None:
    """Horizontal bar of wall-clock share per phase, direct-labeled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = list(breakdown["fractions"])
    fracs = [breakdown["fractions"][p] * 100 for p in phases]
    secs = [breakdown["phases"][p] for p in phases]
    fig, ax = plt.subplots(figsize=(9, 0.65 * max(len(phases), 2) + 1.6),
                           facecolor=_SURFACE)
    bars = ax.barh(phases[::-1], fracs[::-1], color=_UTIL_COLOR, height=0.55)
    for b, frac, sec in zip(bars, fracs[::-1], secs[::-1]):
        ax.annotate(f"{frac:.1f}%  ({sec:.1f}s)",
                    (b.get_width() + 1, b.get_y() + b.get_height() / 2),
                    va="center", fontsize=9, color=_INK)
    ax.set_xlim(0, 105)
    ax.set_xlabel("% of iteration wall-clock", color=_INK_2, fontsize=9)
    _style(ax)
    fig.suptitle(title, fontsize=11, color=_INK, x=0.01, ha="left")
    if subtitle:
        ax.set_title(subtitle, fontsize=8.5, color=_INK_2, loc="left", pad=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png, dpi=300, facecolor=_SURFACE)
    plt.close(fig)
    with open(str(Path(out_png).with_suffix(".json")), "w") as f:
        json.dump(breakdown, f, indent=2)


# ---- artifact policy -----------------------------------------------------------
def require_prediction(predicted: str | None) -> str:
    """The campaign rule, enforced: no measurement without a prior prediction."""
    if not predicted or not predicted.strip():
        raise SystemExit(
            "REFUSING TO RUN: --predicted is empty. State the expected outcome "
            "BEFORE measuring (STAGE6_BENCHMARKS.md governing rule).")
    return predicted.strip()


def write_manifest(artifact_dir, ss: str, what: str, predicted: str,
                   measured: str, mechanism: str, config: dict,
                   rerun_cmd: str, wandb_url: str | None = None) -> Path:
    d = Path(artifact_dir)
    d.mkdir(parents=True, exist_ok=True)
    cfg_hash = hex(abs(hash(json.dumps(config, sort_keys=True))) % 16**8)[2:]
    files = "\n".join(f"- `{p.name}`" for p in sorted(d.iterdir())
                      if p.name != "MANIFEST.md")
    body = f"""# {ss} — {what}

- **Predicted (stated before the run):** {predicted}
- **Measured:** {measured}
- **Mechanism:** {mechanism}
- **Config:** `{json.dumps(config, sort_keys=True)}` (hash `{cfg_hash}`)
- **W&B:** {wandb_url or "(offline / not logged)"}
- **Rerun:** `{rerun_cmd}`
- **Captured:** {time.strftime("%Y-%m-%d %H:%M:%S")}

## Artifacts
{files}
"""
    p = d / "MANIFEST.md"
    p.write_text(body)
    return p
