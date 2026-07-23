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
CREATE TABLE IF NOT EXISTS events(
  rowid INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, session_id TEXT,
  ts REAL, payload TEXT);
"""


class PlatformStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "events.jsonl"
        self.db = sqlite3.connect(str(self.root / "pennydata.db"),
                                  check_same_thread=False)
        self.db.executescript(_SCHEMA)
        self._lock = threading.Lock()

    # -- ingestion -------------------------------------------------------------
    def ingest(self, raw: dict) -> str:
        """Validate + persist one event; returns its kind."""
        ev = parse_event(raw)
        with self._lock:
            with self.log_path.open("a") as f:
                f.write(ev.model_dump_json() + "\n")
            self._apply(ev)
            self.db.execute(
                "INSERT INTO events(kind, session_id, ts, payload) VALUES(?,?,?,?)",
                (ev.kind, ev.session_id, ev.ts, ev.model_dump_json()))
            self.db.commit()
        return ev.kind

    def _apply(self, ev) -> None:
        if ev.kind == "episode_start":
            self.db.execute(
                "INSERT OR REPLACE INTO episodes(session_id, label, policy_ckpt,"
                " brief, started) VALUES(?,?,?,?,?)",
                (ev.session_id, ev.label, str(ev.policy.get("ckpt", "")),
                 ev.brief, ev.ts))
        elif ev.kind == "turn":
            r = parse_agent_action(ev.agent)
            action_kind = r.action.action if r.ok else "invalid"
            self.db.execute(
                "INSERT OR REPLACE INTO turns(session_id, i, agent, observation,"
                " note, action_kind, feedback, ts) VALUES(?,?,?,?,?,?,"
                " (SELECT feedback FROM turns WHERE session_id=? AND i=?), ?)",
                (ev.session_id, ev.i, ev.agent, ev.observation, ev.note,
                 action_kind, ev.session_id, ev.i, ev.ts))
        elif ev.kind == "feedback":
            self.db.execute(
                "UPDATE turns SET feedback=? WHERE session_id=? AND i=?",
                (ev.vote, ev.session_id, ev.i))
        elif ev.kind == "episode_end":
            self.db.execute(
                "UPDATE episodes SET ended=?, verdict=?, violation=?, cart=?"
                " WHERE session_id=?",
                (ev.ts, ev.verdict, int(ev.violation), json.dumps(ev.cart),
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
            for t in ("episodes", "turns", "events"):
                self.db.execute(f"DELETE FROM {t}")
            if self.log_path.exists():
                for line in self.log_path.read_text().splitlines():
                    ev = parse_event(json.loads(line))
                    self._apply(ev)
                    self.db.execute(
                        "INSERT INTO events(kind, session_id, ts, payload)"
                        " VALUES(?,?,?,?)",
                        (ev.kind, ev.session_id, ev.ts, ev.model_dump_json()))
                    n += 1
            self.db.commit()
        return n
