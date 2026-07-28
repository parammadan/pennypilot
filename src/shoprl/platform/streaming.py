"""PennyData streaming layer — Apache Kafka as the store→platform event bus.

Design (the parts worth defending in an interview):
- Topic `pennymart.events`, events KEYED BY session_id — Kafka guarantees
  order within a partition, so keying by session gives per-conversation
  ordering while allowing parallel sessions across partitions.
- Delivery is AT-LEAST-ONCE (producer acks="all", consumer commits after
  apply); the store's ingest is IDEMPOTENT per (session, kind, i) — replayed
  duplicates cannot double-count, so the derived views are effectively-once.
- REPLAY: the broker retains the log, so a fresh consumer group (or
  --from-beginning) rebuilds the entire platform state from the topic — the
  same source-of-truth property the JSONL log gives locally, now with a real
  broker's retention/offset semantics.

The HTTP ingest path stays available; Kafka is an alternative transport into
the same PlatformStore.
"""
from __future__ import annotations

import json

from shoprl.platform.store import PlatformStore

TOPIC = "pennymart.events"


class KafkaEmitter:
    """Producer used by the store side (demo sessions). acks='all' — an event
    is confirmed only when the broker has it; the demo swallows failures after
    one warning (a dead broker must never break a live episode)."""

    def __init__(self, brokers: str, topic: str = TOPIC):
        from kafka import KafkaProducer
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=brokers.split(","), acks="all",
            key_serializer=lambda k: k.encode(),
            value_serializer=lambda v: json.dumps(v).encode())
        self._warned = False

    def emit(self, kind: str, session_id: str, **fields) -> None:
        try:
            import time as _t
            fields.setdefault("ts", _t.time())   # event time, stamped at source
            self.producer.send(self.topic, key=session_id,
                               value={"kind": kind, "session_id": session_id,
                                      **fields})
            self.producer.flush(timeout=3)
        except Exception as e:
            if not self._warned:
                print(f"[kafka] emit failed ({e}) — continuing without")
                self._warned = True

    def close(self) -> None:
        try:
            self.producer.close(timeout=3)
        except Exception:
            pass


def consume_into_store(store: PlatformStore, brokers: str,
                       topic: str = TOPIC, group_id: str = "pennydata",
                       from_beginning: bool = False,
                       max_events: int | None = None) -> int:
    """Consumer loop: topic → PlatformStore. Commits AFTER apply
    (at-least-once); idempotent ingest makes the views effectively-once.
    `max_events` bounds the loop for tests; None = run forever."""
    from kafka import KafkaConsumer
    if from_beginning and group_id == "pennydata":
        # committed offsets belong to the GROUP — auto_offset_reset=earliest
        # only fires for groups with no offsets, so a true replay needs a
        # fresh group id (the log is still there; offsets are just bookmarks).
        # Callers that pass an explicit group manage this themselves (e.g.
        # run_platform shares one replay group across N consumer threads).
        import time as _t
        group_id = f"{group_id}-replay-{int(_t.time())}"
    consumer = KafkaConsumer(
        topic, bootstrap_servers=brokers.split(","), group_id=group_id,
        auto_offset_reset="earliest" if from_beginning else "latest",
        enable_auto_commit=False,
        value_deserializer=lambda b: json.loads(b.decode()))
    n = 0
    try:
        for msg in consumer:
            try:
                store.ingest(msg.value)
            except Exception as e:
                # poison events are logged and skipped, never wedge the stream
                print(f"[kafka] bad event skipped ({e}): {str(msg.value)[:120]}")
            consumer.commit()
            n += 1
            if max_events is not None and n >= max_events:
                break
    finally:
        consumer.close()
    return n
