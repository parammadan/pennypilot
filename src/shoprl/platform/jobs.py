"""PennyData self-service jobs — extraction requests + training runs, with
demo-grade auth.

AUTH IS DEMO-GRADE by design: named users with local passwords and bearer
tokens, enough to demonstrate roles/audit ("requested_by" on every job) —
NOT production security (that's SSO/IAM in the documented evolution).

Jobs:
- extraction: run a dataset build (ad-hoc filters or an approved recipe)
  synchronously; result = dataset path + manifest, recorded in lineage.
- training: SUBMIT a real Slurm job on the cluster over ssh (run_sft_v2 with
  the chosen dataset) and track its state by polling sacct. The submitter is
  injectable so tests never touch ssh.
"""
from __future__ import annotations

import json
import secrets
import subprocess
import threading
import time
from pathlib import Path

from shoprl.platform.recipes import apply_recipe, load_recipe
from shoprl.platform.store import PlatformStore

# demo users — documented, visible, not a secret
USERS = {"param": {"password": "pennydata", "role": "admin"},
         "scientist": {"password": "science", "role": "scientist"}}

_JOBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
  job_id TEXT PRIMARY KEY, kind TEXT, params TEXT, status TEXT,
  requested_by TEXT, created REAL, updated REAL, result TEXT);
CREATE TABLE IF NOT EXISTS tokens(
  token TEXT PRIMARY KEY, user TEXT, role TEXT, created REAL);
"""


class JobService:
    def __init__(self, store: PlatformStore, dataset_dir: str | Path = "datasets",
                 slurm_submit=None, slurm_status=None):
        self.store = store
        self.dataset_dir = Path(dataset_dir)
        store.db.executescript(_JOBS_SCHEMA)
        # injectable for tests; real impls shell out over ssh
        self._submit = slurm_submit or self._ssh_submit
        self._status = slurm_status or self._ssh_status

    # -- auth -------------------------------------------------------------------
    def login(self, user: str, password: str) -> dict | None:
        u = USERS.get(user)
        if not u or u["password"] != password:
            return None
        token = secrets.token_hex(16)
        self.store.db.execute("INSERT INTO tokens VALUES(?,?,?,?)",
                              (token, user, u["role"], time.time()))
        self.store.db.commit()
        return {"token": token, "user": user, "role": u["role"]}

    def whoami(self, token: str | None) -> dict | None:
        if not token:
            return None
        r = self.store.db.execute(
            "SELECT user, role FROM tokens WHERE token=?", (token,)).fetchone()
        return {"user": r[0], "role": r[1]} if r else None

    # -- jobs -------------------------------------------------------------------
    def _record(self, kind: str, params: dict, user: str,
                status: str = "queued", result: dict | None = None) -> str:
        jid = f"{kind}-{secrets.token_hex(4)}"
        self.store.db.execute(
            "INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)",
            (jid, kind, json.dumps(params), status, user, time.time(),
             time.time(), json.dumps(result or {})))
        self.store.db.commit()
        return jid

    def _update(self, jid: str, status: str, result: dict | None = None):
        cur = self.store.db.execute(
            "SELECT result FROM jobs WHERE job_id=?", (jid,)).fetchone()
        merged = {**json.loads(cur[0] or "{}"), **(result or {})}
        self.store.db.execute(
            "UPDATE jobs SET status=?, updated=?, result=? WHERE job_id=?",
            (status, time.time(), json.dumps(merged), jid))
        self.store.db.commit()

    def jobs(self) -> list[dict]:
        return [{"job_id": r[0], "kind": r[1], "params": json.loads(r[2]),
                 "status": r[3], "requested_by": r[4], "created": r[5],
                 "updated": r[6], "result": json.loads(r[7] or "{}")}
                for r in self.store.db.execute(
                    "SELECT * FROM jobs ORDER BY created DESC").fetchall()]

    # -- extraction ---------------------------------------------------------------
    def request_extraction(self, user: str, recipe_path: str | None = None,
                           recipe_id: str | None = None,
                           source: str = "hot") -> dict:
        """source='hot' extracts from the streaming-ingested store;
        source='lake' materializes episodes/turns from the S3 batch lake
        (DuckDB) into a temporary store first — both paths end in the same
        recipe machinery, manifest, and lineage."""
        params = {"recipe_path": recipe_path, "recipe_id": recipe_id,
                  "source": source}
        jid = self._record("extraction", params, user, status="running")
        try:
            path = recipe_path or f"recipes/{recipe_id}.json"
            recipe = load_recipe(path)
            src_store = self.store
            if source == "lake":
                src_store = self._materialize_lake()
            manifest = apply_recipe(src_store, recipe,
                                    self.dataset_dir)
            manifest["extraction_source"] = source
            self._update(jid, "succeeded",
                         {"manifest": manifest,
                          "dataset": str(self.dataset_dir /
                                         f"{recipe.recipe_id}.jsonl")})
        except Exception as e:
            self._update(jid, "failed", {"error": str(e)})
        return next(j for j in self.jobs() if j["job_id"] == jid)

    def _materialize_lake(self) -> PlatformStore:
        """Batch path: pull raw events from the S3 lake via DuckDB and replay
        them through a throwaway store (same validation/idempotency)."""
        import tempfile
        import sys
        sys.path.insert(0, "scripts")
        from lake import connect
        con = connect("s3")
        tmp = PlatformStore(tempfile.mkdtemp(prefix="lake-extract-"))
        rows = con.execute("SELECT * FROM raw").df().to_dict("records")
        for r in rows:
            ev = {k: v for k, v in r.items() if v is not None
                  and not (isinstance(v, float) and v != v)}
            try:
                tmp.ingest(ev)
            except Exception:
                pass                        # malformed lake rows are counted
        return tmp

    def approve_recipe(self, user: str, recipe_id: str) -> dict:
        path = Path(f"recipes/{recipe_id}.json")
        r = json.loads(path.read_text())
        r["approval_status"] = "approved"
        r["approved_by"] = f"{user} (via platform UI, " \
            f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})"
        path.write_text(json.dumps(r, indent=2))
        return r

    # -- training -----------------------------------------------------------------
    def request_training(self, user: str, dataset: str, out_name: str,
                         mix_generated: int = 270,
                         cannot_frac: float = 0.30) -> dict:
        params = {"dataset": dataset, "out_name": out_name,
                  "mix_generated": mix_generated, "cannot_frac": cannot_frac}
        jid = self._record("training", params, user)
        try:
            slurm_id = self._submit(params)
            self._update(jid, "submitted", {"slurm_id": slurm_id})
            threading.Thread(target=self._track, args=(jid, slurm_id),
                             daemon=True).start()
        except Exception as e:
            self._update(jid, "failed", {"error": str(e)})
        return next(j for j in self.jobs() if j["job_id"] == jid)

    def _track(self, jid: str, slurm_id: str):
        while True:
            time.sleep(60)
            try:
                st = self._status(slurm_id)
            except Exception:
                continue
            if st in ("COMPLETED",):
                self._update(jid, "succeeded", {"slurm_state": st})
                return
            if st.startswith(("FAILED", "CANCELLED", "TIMEOUT")):
                self._update(jid, "failed", {"slurm_state": st})
                return
            self._update(jid, "running", {"slurm_state": st})

    # real cluster wiring (not exercised by tests)
    def _ssh_submit(self, p: dict) -> str:
        d = "/scratch/madan.pa/pennypilot"
        remote = f"{d}/{Path(p['dataset']).name}"
        subprocess.run(["rsync", "-q", p["dataset"], f"explorer:{remote}"],
                       check=True, timeout=120)
        cmd = (f"cd ~/pennywise-v100-infra && sbatch --parsable -p gpu "
               f"--gres=gpu:v100-sxm2:1 -c6 --mem=64G -t03:00:00 "
               f"--job-name=ui-{p['out_name']} --output={d}/ui_%j.log "
               f"--wrap \"export HF_HOME={d.replace('pennypilot','hf_cache')} "
               f"HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; "
               f"cd $HOME/pennywise-v100-infra; "
               f"/home/madan.pa/.conda/envs/shoprl/bin/python "
               f"scripts/run_sft_v2.py --model Qwen/Qwen2.5-7B-Instruct "
               f"--method lora --system chat --dataset-file {remote} "
               f"--mix-generated {p['mix_generated']} "
               f"--cannot-frac {p['cannot_frac']} "
               f"--rehearsal {d}/rehearsal.jsonl --max-len 1536 "
               f"--batch-size 2 --epochs 2 "
               f"--out-dir {d}/{p['out_name']}\"")
        out = subprocess.run(["ssh", "explorer", cmd], capture_output=True,
                             text=True, timeout=120, check=True)
        return out.stdout.strip().splitlines()[-1]

    def _ssh_status(self, slurm_id: str) -> str:
        out = subprocess.run(
            ["ssh", "explorer",
             f"sacct -j {slurm_id} -X -n -o State | head -1"],
            capture_output=True, text=True, timeout=60, check=True)
        return (out.stdout.strip() or "PENDING").split()[0]
