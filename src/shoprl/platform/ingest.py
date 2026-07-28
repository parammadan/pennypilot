"""PennyData ingest + API service (stdlib http.server, same idiom as
serve_policy.py — the console polls it, the store POSTs to it).

  POST /events   one event dict or a list of them
  GET  /         the console UI
  GET  /feed?since=<rowid>
  GET  /stats
  GET  /query?sql=SELECT ...
  GET  /export?label=&feedback=&include_violations=1
  GET  /health
"""
from __future__ import annotations

import json
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from shoprl.platform.behavior import (alerts, data_quality,
                                       detect_anomalies, funnel_by, metrics,
                                       session_features, slice_report,
                                       timeseries)
from shoprl.platform.console import CONSOLE_HTML
from shoprl.platform.export import build_dataset
from shoprl.platform.quality import stats
from shoprl.platform.store import PlatformStore


def make_handler(store: PlatformStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/events":
                return self._json({"error": "unknown endpoint"}, 404)
            try:
                raw = json.loads(self.rfile.read(
                    int(self.headers.get("Content-Length", 0))))
                events = raw if isinstance(raw, list) else [raw]
                kinds = [store.ingest(e) for e in events]
                self._json({"ok": True, "ingested": len(kinds)})
            except Exception as e:
                self._json({"error": str(e)}, 400)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    self._send(200, CONSOLE_HTML.encode(), "text/html")
                elif u.path == "/health":
                    self._json({"ok": True, "root": str(store.root)})
                elif u.path == "/feed":
                    self._json(store.feed(since_rowid=int(q.get("since", 0))))
                elif u.path == "/stats":
                    self._json(stats(store))
                elif u.path == "/query":
                    self._json(store.query(q.get("sql", "")))
                elif u.path == "/behavior":
                    sess = session_features(store)
                    self._json({"data_quality": data_quality(store),
                                "metrics": metrics(sess)})
                elif u.path == "/slices":
                    sess = session_features(store)
                    self._json(slice_report(
                        sess, metric=q.get("metric", "abandoned"),
                        min_support=int(q.get("min_support", 30)),
                        top=int(q.get("top", 10))))
                elif u.path == "/funnel":
                    self._json(funnel_by(None, store,
                                         label=q.get("label") or None))
                elif u.path == "/alerts":
                    self._json(alerts(store))
                elif u.path == "/timeseries":
                    ser = timeseries(store, metric=q.get("metric", "abandoned"),
                                     bucket_s=int(q.get("bucket", 300)))
                    self._json({"series": ser,
                                "anomalies": detect_anomalies(ser)})
                elif u.path == "/sessions":
                    # failure-analysis API (Lesson 7): hand the underlying
                    # sessions matching a behavioral condition to a human
                    sess = session_features(store)
                    for k in ("label", "turn_bucket"):
                        if q.get(k):
                            sess = [x for x in sess if str(x[k]) == q[k]]
                    for k in ("abandoned", "violation", "reformulated",
                              "repeated", "carted"):
                        if q.get(k):
                            want = q[k] in ("1", "true")
                            sess = [x for x in sess if x[k] == want]
                    sids = [x["session_id"] for x in sess[:int(q.get("limit", 20))]]
                    detail = []
                    for sid in sids:
                        turns = store.db.execute(
                            "SELECT i, agent, observation, note, feedback"
                            " FROM turns WHERE session_id=? ORDER BY i",
                            (sid,)).fetchall()
                        detail.append({"session_id": sid, "turns": [
                            {"i": t[0], "agent": t[1], "observation": t[2],
                             "note": t[3], "feedback": t[4]} for t in turns]})
                    self._json({"matched": len(sess), "returned": len(detail),
                                "sessions": detail})
                elif u.path == "/export":
                    self._json(build_dataset(
                        store, label=q.get("label") or None,
                        feedback=q.get("feedback") or None,
                        exclude_violations=q.get("include_violations") != "1"))
                else:
                    self._json({"error": "unknown endpoint"}, 404)
            except Exception as e:
                self._json({"error": str(e)}, 400)

    return Handler


def serve(root: str, port: int = 8770) -> None:
    store = PlatformStore(root)
    httpd = HTTPServer(("0.0.0.0", port), make_handler(store))
    print(f"[pennydata] listening on {socket.gethostname()}:{port} "
          f"(store: {store.root})", flush=True)
    httpd.serve_forever()
