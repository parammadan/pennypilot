"""PennyData analyst workbench — DuckDB SQL over the event lake.

The HTML console is the OPS view; this is the ANALYSIS view: full SQL
(joins, windows, CTEs) directly over the S3 cold path and/or the local hot
log — no server, no ETL, the Hive date= partitioning is queried as-is.

    python scripts/lake.py --report funnel            # canned analyses
    python scripts/lake.py --report cohorts
    python scripts/lake.py --report slices
    python scripts/lake.py "SELECT kind, COUNT(*) FROM raw GROUP BY 1"
    python scripts/lake.py --source local ...         # hot log instead of S3

Views installed for every session:
  raw       every event (kind, session_id, ts, + kind-specific columns)
  episodes  one row per episode (label, brief, verdict, violation, cart)
  turns     agent turns (agent text, observation, note)
  ui        clickstream ticks (type, target)
"""
import argparse

import duckdb

BUCKET = "pennydata-771965334314-us-east-2"
S3_GLOB = f"s3://{BUCKET}/pennydata/events/date=*/part-*.jsonl"
LOCAL = "~/pennydata/events.jsonl"

REPORTS = {
    "funnel": """
        WITH stages(stage, ord, action) AS (VALUES
          ('engaged', 1, 'ask_user'), ('searched', 2, 'search'),
          ('selected', 3, 'select_product'),
          ('permission', 4, 'request_cart_permission'),
          ('carted', 5, 'add_to_cart'))
        SELECT stage, COUNT(DISTINCT t.session_id) AS sessions,
               ROUND(COUNT(DISTINCT t.session_id) * 1.0 /
                 NULLIF(LAG(COUNT(DISTINCT t.session_id))
                        OVER (ORDER BY ord), 0), 3) AS conv_from_prev
        FROM stages s JOIN turns t
          ON t.agent LIKE '%"action": "' || s.action || '"%'
        GROUP BY stage, ord ORDER BY ord""",
    "cohorts": """
        SELECT e.label, COUNT(*) AS episodes,
               ROUND(AVG(CASE WHEN e.cart != '[]' THEN 1 ELSE 0 END), 3)
                 AS cart_rate,
               ROUND(AVG(e.violation::INT), 4) AS violation_rate,
               ROUND(AVG(t.n_turns), 1) AS avg_turns
        FROM episodes e JOIN (SELECT session_id, COUNT(*) n_turns
                              FROM turns GROUP BY 1) t USING (session_id)
        GROUP BY 1 ORDER BY episodes DESC""",
    "slices": """
        WITH s AS (
          SELECT e.session_id, e.label,
                 CASE WHEN t.n <= 3 THEN '1-3' WHEN t.n <= 6 THEN '4-6'
                      ELSE '7+' END AS turn_bucket,
                 CASE WHEN e.cart = '[]' THEN 1 ELSE 0 END AS abandoned
          FROM episodes e JOIN (SELECT session_id, COUNT(*) n FROM turns
                                GROUP BY 1) t USING (session_id))
        SELECT label, turn_bucket, COUNT(*) AS n,
               ROUND(AVG(abandoned), 3) AS abandonment,
               ROUND(AVG(abandoned) - (SELECT AVG(abandoned) FROM s), 3)
                 AS deviation
        FROM s GROUP BY 1, 2 HAVING COUNT(*) >= 30
        ORDER BY ABS(AVG(abandoned) - (SELECT AVG(abandoned) FROM s)) DESC
        LIMIT 12""",
    "hourly": """
        SELECT strftime(to_timestamp(ts), '%Y-%m-%d %H:00') AS hour,
               COUNT(DISTINCT session_id) AS sessions,
               SUM(CASE WHEN kind = 'ui' THEN 1 ELSE 0 END) AS ui_ticks
        FROM raw GROUP BY 1 ORDER BY 1""",
}


def connect(source: str = "s3") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    if source == "s3":
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("CREATE SECRET (TYPE S3, PROVIDER CREDENTIAL_CHAIN);")
        src = f"read_json_auto('{S3_GLOB}', union_by_name=true)"
    else:
        src = f"read_json_auto('{LOCAL}', union_by_name=true)"
    con.execute(f"CREATE VIEW raw AS SELECT * FROM {src}")
    con.execute("""
        CREATE VIEW episodes AS
        SELECT s.session_id, s.label, s.brief, s.started,
               e.verdict, e.violation, e.cart
        FROM (SELECT session_id, label, brief, ts AS started FROM raw
              WHERE kind = 'episode_start') s
        LEFT JOIN (SELECT session_id, verdict,
                          COALESCE(violation, false) AS violation,
                          COALESCE(cart::VARCHAR, '[]') AS cart
                   FROM raw WHERE kind = 'episode_end') e
          USING (session_id)""")
    con.execute("CREATE VIEW turns AS SELECT session_id, i, agent,"
                " observation, note, ts FROM raw WHERE kind = 'turn'")
    con.execute("CREATE VIEW ui AS SELECT session_id, type, target, ts"
                " FROM raw WHERE kind = 'ui'")
    return con


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sql", nargs="?", default=None)
    ap.add_argument("--report", choices=sorted(REPORTS))
    ap.add_argument("--source", choices=["s3", "local"], default="s3")
    args = ap.parse_args()
    con = connect(args.source)
    q = REPORTS[args.report] if args.report else args.sql
    if not q:
        ap.error("give SQL or --report")
    print(con.execute(q).df().to_string(index=False))
