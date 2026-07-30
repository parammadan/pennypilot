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

from shoprl.platform.attribution import attribute_all, attribute_session
from shoprl.platform.milestones import friction_metrics, milestone_funnel
from shoprl.platform.behavior import (alerts, data_quality,
                                       detect_anomalies, funnel_by, metrics,
                                       session_features, slice_report,
                                       timeseries)
from shoprl.platform.console import CONSOLE_HTML
from shoprl.platform.export import build_dataset
from shoprl.platform.quality import stats
from shoprl.platform.jobs import JobService
from shoprl.platform.store import PlatformStore


def make_handler(store: PlatformStore, jobsvc: JobService | None = None):
    jobsvc = jobsvc or JobService(store)
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

        def _auth(self):
            tok = (self.headers.get("Authorization") or "").replace(
                "Bearer ", "")
            return jobsvc.whoami(tok)

        def do_POST(self):
            u = urlparse(self.path)
            try:
                body = json.loads(self.rfile.read(
                    int(self.headers.get("Content-Length", 0)) or 2) or "{}")
            except Exception:
                body = {}
            try:
                if u.path == "/events":
                    events = body if isinstance(body, list) else [body]
                    kinds = [store.ingest(e) for e in events]
                    return self._json({"ok": True, "ingested": len(kinds)})
                if u.path == "/login":
                    r = jobsvc.login(body.get("user", ""),
                                     body.get("password", ""))
                    return self._json(r or {"error": "bad credentials"},
                                      200 if r else 401)
                # everything below requires a logged-in user
                me = self._auth()
                if not me:
                    return self._json({"error": "login required"}, 401)
                if u.path == "/extract":
                    return self._json(jobsvc.request_extraction(
                        me["user"], recipe_id=body.get("recipe_id"),
                        source=body.get("source", "hot")))
                if u.path == "/train":
                    return self._json(jobsvc.request_training(
                        me["user"], dataset=body["dataset"],
                        out_name=body["out_name"],
                        mix_generated=int(body.get("mix_generated", 270)),
                        cannot_frac=float(body.get("cannot_frac", 0.30))))
                if u.path == "/recipes/approve":
                    if me["role"] != "admin":
                        return self._json({"error": "admin only"}, 403)
                    return self._json(jobsvc.approve_recipe(
                        me["user"], body["recipe_id"]))
                return self._json({"error": "unknown endpoint"}, 404)
            except Exception as e:
                self._json({"error": str(e)}, 400)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html") or u.path.startswith("/assets/"):
                    import os
                    dist = os.path.join(os.path.dirname(__file__),
                                        "..", "..", "..", "dashboard", "dist")
                    rel = "index.html" if u.path in ("/", "/index.html")                         else u.path.lstrip("/")
                    fp = os.path.normpath(os.path.join(dist, rel))
                    if os.path.normpath(dist) in fp and os.path.isfile(fp):
                        ctype = ("text/html" if fp.endswith(".html") else
                                 "text/javascript" if fp.endswith(".js") else
                                 "text/css" if fp.endswith(".css") else
                                 "application/octet-stream")
                        self._send(200, open(fp, "rb").read(), ctype)
                    else:
                        self._send(200, CONSOLE_HTML.encode(), "text/html")
                elif u.path == "/legacy":
                    self._send(200, CONSOLE_HTML.encode(), "text/html")
                elif u.path == "/health":
                    self._json({"ok": True, "root": str(store.root)})
                elif u.path == "/feed":
                    self._json(store.feed(since_rowid=int(q.get("since", 0))))
                elif u.path == "/stats":
                    self._json(stats(store))
                elif u.path == "/query":
                    # governance: raw SQL is ADMIN-ONLY; everyone else uses
                    # form-driven extraction requests
                    me = self._auth()
                    if not me or me["role"] != "admin":
                        return self._json(
                            {"error": "raw SQL is admin-only — use "
                                      "extraction requests"}, 403)
                    self._json(store.query(q.get("sql", "")))
                elif u.path == "/me":
                    self._json(self._auth() or {})
                elif u.path == "/jobs":
                    self._json(jobsvc.jobs())
                elif u.path == "/privacy":
                    rows = store.db.execute(
                        "SELECT pii_type, SUM(n) FROM privacy_log GROUP BY 1"
                    ).fetchall()
                    self._json({"redactions_by_type": dict(rows),
                                "total": sum(r[1] for r in rows),
                                "note": "synthetic data should redact ZERO — "
                                        "nonzero is a data-quality alarm"})
                elif u.path == "/logs":
                    logins = store.db.execute(
                        "SELECT user, role, created FROM tokens"
                        " ORDER BY created DESC LIMIT 50").fetchall()
                    self._json({"jobs": jobsvc.jobs()[:50],
                                "logins": [{"user": l[0], "role": l[1],
                                            "ts": l[2]} for l in logins]})
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
                elif u.path == "/funnel2":
                    self._json(milestone_funnel(store))
                elif u.path == "/friction":
                    self._json(friction_metrics(store))
                elif u.path == "/attribution":
                    sid = q.get("session_id")
                    self._json(attribute_session(store, sid) if sid
                               else attribute_all(store))
                elif u.path == "/lineage":
                    try:
                        rows = store.db.execute(
                            "SELECT recipe_id, dataset_sha, created, manifest"
                            " FROM lineage ORDER BY created DESC").fetchall()
                    except Exception:
                        rows = []
                    self._json([{"recipe_id": r[0], "dataset_sha": r[1],
                                 "created": r[2],
                                 "manifest": json.loads(r[3])} for r in rows])
                elif u.path == "/recipes":
                    import glob as _g
                    out = []
                    for f in sorted(_g.glob("recipes/*.json")):
                        try:
                            out.append(json.loads(open(f).read()))
                        except Exception:
                            pass
                    self._json(out)
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
