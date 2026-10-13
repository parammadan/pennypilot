"""Run reproducibility: one seed for every RNG, one manifest per run.

Scenario/catalog generation is already seeded by string-keyed `random.Random`
instances; this covers the rest (torch sampling during rollouts, dropout,
data-loader shuffles, numpy). GPU kernels can still be nondeterministic, so
seeded runs are *repeatable in distribution*, not bit-identical — the paper
reports mean ± std over seeds for exactly that reason.
"""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def write_manifest(out_dir: str, args, **extra) -> str:
    """`<out_dir>/manifest.json`: argv, parsed args, git sha + dirty flag,
    SLURM job id, host, versions. Written at run START so failed runs are
    recorded too."""
    versions = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "peft"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            pass
    man = {
        "argv": sys.argv,
        "args": vars(args),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": platform.node(),
        "started_unix": int(time.time()),
        "versions": versions,
        **extra,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(man, f, indent=2, default=str)
    return path
