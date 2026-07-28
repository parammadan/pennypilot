"""PennyData store — append-only JSONL event log + derived SQLite views.

The JSONL log is the source of truth (rebuild the DB by replaying it); SQLite
gives the console self-service SQL. Writes are idempotent per (session, kind,
i) where that makes sense, so a re-sent event never double-counts.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from shoprl.actions import parse_agent_action
from shoprl.platform.events import parse_event

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes(
  session_id TEXT PRIMARY KEY, label TEXT, policy_ckpt TEXT, brief TEXT,
  started REAL, ended REAL, verdict TEXT, violation INTEGER, cart TEXT);
CREATE TABLE IF NOT EXISTS turns(
  session_id TEXT, i INTEGER, agent TEXT, observation TEXT, note TEXT,
  action_kind TEXT, feedback TEXT, ts REAL,
  PRIMARY KEY(session_id, i));
CREATE TABLE IF NOT EXISTS ui_events(
  session_id TEXT, type TEXT, target TEXT, meta TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS semantic_events(
  session_id TEXT, event_id TEXT UNIQUE, type TEXT, turn_index INTEGER,
  attributes TEXT, source TEXT, model_version TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS events(
  rowid INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, session_id TEXT,
  ts REAL, ingest_ts REAL, event_id TEXT, payload TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_eid ON events(event_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ui_eid ON ui_events(event_id);
"""


class PlatformStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "events.jsonl"
        self.db = sqlite3.connect(str(self.root / "pennydata.db"),
                                  check_same_thread=False)
        self.db.executescript(
            _SCHEMA.split("CREATE UNIQUE INDEX")[0])   # tables first
        for mig in ("ALTER TABLE turns ADD COLUMN latency_ms REAL",
                    "ALTER TABLE episodes ADD COLUMN goal TEXT",
                    "ALTER TABLE episodes ADD COLUMN outcome TEXT",
                    "ALTER TABLE episodes ADD COLUMN model_version TEXT",
                    "ALTER TABLE episodes ADD COLUMN scenario_family TEXT",
                    "ALTER TABLE events ADD COLUMN ingest_ts REAL",
                    "ALTER TABLE events ADD COLUMN event_id TEXT",
                    "ALTER TABLE ui_events ADD COLUMN event_id TEXT"):
            try:
                self.db.execute(mig)
            except sqlite3.OperationalError:
                pass                   # column already there (or fresh schema)
        self.db.executescript(_SCHEMA)
        self._lock = threading.Lock()

    # -- ingestion -------------------------------------------------------------
    def ingest(self, raw: dict) -> str:
        """Validate + persist one event; returns its kind, or "duplicate" —
        an event_id seen before is skipped ENTIRELY (idempotent under Kafka
        redelivery and log re-ingestion)."""
        import time as _t
        ev = parse_event(raw)
        with self._lock:
            eid = getattr(ev, "event_id", None)
            if eid and self.db.execute(
                    "SELECT 1 FROM events WHERE event_id=?", (eid,)).fetchone():
                return "duplicate"
            with self.log_path.open("a") as f:
                f.write(ev.model_dump_json() + "\n")
            self._apply(ev)
            self.db.execute(
                "INSERT INTO events(kind, session_id, ts, ingest_ts, event_id,"
                " payload) VALUES(?,?,?,?,?,?)",
                (ev.kind, ev.session_id, ev.ts, _t.time(), eid,
                 ev.model_dump_json()))
            self.db.commit()
        return ev.kind

    def _apply(self, ev) -> None:
        if ev.kind == "episode_start":
            goal = getattr(ev, "goal", None)
            self.db.execute(
                "INSERT OR REPLACE INTO episodes(session_id, label, policy_ckpt,"
                " brief, started, goal, model_version, scenario_family)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (ev.session_id, ev.label, str(ev.policy.get("ckpt", "")),
                 ev.brief, ev.ts,
                 goal.model_dump_json() if goal is not None else None,
                 getattr(ev, "model_version", "") or ev.label,
                 getattr(ev, "scenario_family", "")))
        elif ev.kind == "turn":
            r = parse_agent_action(ev.agent)
            action_kind = r.action.action if r.ok else "invalid"
            self.db.execute(
                "INSERT OR REPLACE INTO turns(session_id, i, agent, observation,"
                " note, action_kind, feedback, latency_ms, ts) VALUES(?,?,?,?,?,?,"
                " (SELECT feedback FROM turns WHERE session_id=? AND i=?), ?, ?)",
                (ev.session_id, ev.i, ev.agent, ev.observation, ev.note,
                 action_kind, ev.session_id, ev.i, ev.latency_ms, ev.ts))
        elif ev.kind == "feedback":
            self.db.execute(
                "UPDATE turns SET feedback=? WHERE session_id=? AND i=?",
                (ev.vote, ev.session_id, ev.i))
        elif ev.kind == "ui":
            self.db.execute(
                "INSERT OR IGNORE INTO ui_events(session_id, type, target,"
                " meta, ts, event_id) VALUES(?,?,?,?,?,?)",
                (ev.session_id, ev.type, ev.target, json.dumps(ev.meta),
                 ev.ts, getattr(ev, "event_id", None)))
        elif ev.kind == "semantic":
            self.db.execute(
                "INSERT OR IGNORE INTO semantic_events(session_id, event_id,"
                " type, turn_index, attributes, source, model_version, ts)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (ev.session_id, ev.event_id, ev.type, ev.turn_index,
                 json.dumps(ev.attributes), ev.source, ev.model_version, ev.ts))
        elif ev.kind == "episode_end":
            out = getattr(ev, "outcome", None)
            self.db.execute(
                "UPDATE episodes SET ended=?, verdict=?, violation=?, cart=?,"
                " outcome=? WHERE session_id=?",
                (ev.ts, ev.verdict, int(ev.violation), json.dumps(ev.cart),
                 out.model_dump_json() if out is not None else None,
                 ev.session_id))

    # -- reads -----------------------------------------------------------------
    def feed(self, since_rowid: int = 0, limit: int = 100) -> list[dict]:
        cur = self.db.execute(
            "SELECT rowid, kind, session_id, ts, payload FROM events"
            " WHERE rowid > ? ORDER BY rowid DESC LIMIT ?", (since_rowid, limit))
        return [{"rowid": r[0], "kind": r[1], "session_id": r[2], "ts": r[3],
                 "payload": json.loads(r[4])} for r in cur.fetchall()]

    def query(self, sql: str, limit: int = 200) -> dict:
        """Self-service SQL — SELECT-only, row-limited, read-only connection."""
        if not sql.strip().lower().startswith("select") or ";" in sql.strip()[:-1]:
            raise ValueError("SELECT-only queries (single statement)")
        cur = self.db.execute(sql.rstrip().rstrip(";") + f" LIMIT {int(limit)}")
        cols = [d[0] for d in cur.description]
        return {"columns": cols, "rows": [list(r) for r in cur.fetchall()]}

    def rebuild_from_log(self) -> int:
        """Replay events.jsonl into a fresh derived view (source-of-truth demo)."""
        n = 0
        with self._lock:
            for t in ("episodes", "turns", "ui_events", "semantic_events",
                      "events"):
                self.db.execute(f"DELETE FROM {t}")
            if self.log_path.exists():
                for line in self.log_path.read_text().splitlines():
                    ev = parse_event(json.loads(line))
                    self._apply(ev)
                    self.db.execute(
                        "INSERT OR IGNORE INTO events(kind, session_id, ts,"
                        " event_id, payload) VALUES(?,?,?,?,?)",
                        (ev.kind, ev.session_id, ev.ts,
                         getattr(ev, "event_id", None), ev.model_dump_json()))
                    n += 1
            self.db.commit()
        return n
