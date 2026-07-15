"""Metrics data-source abstraction.

The dashboard reads through a `MetricsSource`, so the SAME dashboard code serves
replay, live-tail, and future cloud metrics — connecting live data later is
choosing a different source, not a rebuild:

  - StaticFileSource : a finished metrics.jsonl (REPLAY — primary, demoable now)
  - LiveTailSource   : a growing metrics.jsonl from a live training run
                       (re-reads each poll) — pre-wired, not yet pointed at a run
  - AWSMetricsSource : future A10G rollout throughput / GPU util from CloudWatch
                       — pre-wired STUB, reports unavailable ("awaiting live data")
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class MetricsSource(Protocol):
    name: str
    def read(self) -> list[dict]: ...
    def available(self) -> bool: ...


def _read_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


@dataclass
class StaticFileSource:
    """A finished metrics.jsonl replayed as a static snapshot (primary path)."""
    path: str
    name: str = "static"

    def read(self) -> list[dict]:
        return _read_jsonl(self.path)

    def available(self) -> bool:
        return Path(self.path).exists()


@dataclass
class LiveTailSource:
    """A live training run's growing metrics.jsonl. Same read contract; the
    dashboard re-polls it for live updates. Drop-in for a running job — no
    dashboard changes needed to go live."""
    path: str
    name: str = "live-tail"

    def read(self) -> list[dict]:
        return _read_jsonl(self.path)

    def available(self) -> bool:
        return Path(self.path).exists()


@dataclass
class AWSMetricsSource:
    """STUB for future A10G rollout throughput / GPU util (CloudWatch). Pre-wired
    so the system-health panels connect by swapping this in; until implemented it
    reports unavailable so those panels render as 'awaiting live data'."""
    namespace: str = "Pennywise/Rollout"
    name: str = "aws-cloudwatch"

    def read(self) -> list[dict]:
        return []  # awaiting live data — Step 2/3 wires this to CloudWatch

    def available(self) -> bool:
        return False
