"""PennyData cold path — Kafka → S3 archival sink (tiered retention).

A SECOND, independent consumer group on the same topic: the broker fans out
to the analytics store (hot path, group `pennydata`) and to object storage
(cold path, group `pennydata-s3`) without either knowing about the other —
the fan-out is the point of a log-based bus.

Layout: s3://<bucket>/pennydata/events/date=YYYY-MM-DD/part-<epoch>-<seq>.jsonl
- date= Hive-style partitioning (Athena/Spark can query it as-is later)
- parts flush on batch size OR age, so a quiet stream still lands durably
- at-least-once (commit after successful put) + append-only immutable parts:
  duplicate parts are possible after a crash, duplicate EVENTS within the
  analytics view are not (the store dedupes on replay)
"""
from __future__ import annotations

import json
import time


class S3Archiver:
    def __init__(self, bucket: str, prefix: str = "pennydata/events",
                 batch_size: int = 500, max_age_s: float = 30.0, s3=None):
        import boto3
        self.s3 = s3 or boto3.client("s3")
        self.bucket, self.prefix = bucket, prefix
        self.batch_size, self.max_age_s = batch_size, max_age_s
        self.buf: list[str] = []
        self.opened = time.time()
        self.seq = 0
        self.parts_written = 0
        self.events_written = 0

    def add(self, event: dict) -> bool:
        """Buffer one event; returns True if this call flushed a part."""
        self.buf.append(json.dumps(event))
        if len(self.buf) >= self.batch_size or \
                time.time() - self.opened >= self.max_age_s:
            self.flush()
            return True
        return False

    def flush(self) -> str | None:
        if not self.buf:
            return None
        day = time.strftime("%Y-%m-%d", time.gmtime())
        key = (f"{self.prefix}/date={day}/"
               f"part-{int(self.opened)}-{self.seq:05d}.jsonl")
        self.s3.put_object(Bucket=self.bucket, Key=key,
                           Body=("\n".join(self.buf) + "\n").encode())
        self.parts_written += 1
        self.events_written += len(self.buf)
        self.buf = []
        self.opened = time.time()
        self.seq += 1
        return key


def archive_from_kafka(bucket: str, brokers: str,
                       topic: str = "pennymart.events",
                       group_id: str = "pennydata-s3",
                       from_beginning: bool = False,
                       max_events: int | None = None,
                       batch_size: int = 500) -> S3Archiver:
    """Consume the topic into S3 parts; commits offsets only after the part
    holding those events is durably in S3 (at-least-once)."""
    from kafka import KafkaConsumer
    if from_beginning:
        group_id = f"{group_id}-replay-{int(time.time())}"
    consumer = KafkaConsumer(
        topic, bootstrap_servers=brokers.split(","), group_id=group_id,
        auto_offset_reset="earliest" if from_beginning else "latest",
        enable_auto_commit=False, consumer_timeout_ms=15_000,
        value_deserializer=lambda b: json.loads(b.decode()))
    arch = S3Archiver(bucket, batch_size=batch_size)
    n = 0
    try:
        for msg in consumer:
            if arch.add(msg.value):
                consumer.commit()      # events are durable in S3 — safe point
            n += 1
            if max_events is not None and n >= max_events:
                break
        key = arch.flush()
        if key:
            consumer.commit()
        print(f"[s3-sink] {arch.events_written} events in "
              f"{arch.parts_written} parts -> s3://{bucket}/{arch.prefix}/")
    finally:
        consumer.close()
    return arch
