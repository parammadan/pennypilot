"""PennyData — the open data platform fed by the PennyMart store.

Interactions in, datasets out: event ingestion (events/store), quality
intelligence (quality), dataset generation with validation (export), and a
live console (console/ingest). The flywheel in one package:
store → platform → training data → better model → store.
"""
from shoprl.platform.events import (EpisodeEnd, EpisodeStart, Feedback, Turn,
                                    parse_event)
from shoprl.platform.store import PlatformStore

__all__ = ["EpisodeStart", "Turn", "Feedback", "EpisodeEnd", "parse_event",
           "PlatformStore"]
