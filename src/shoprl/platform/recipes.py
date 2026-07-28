"""Versioned data recipes — findings → reproducible, approved datasets.

Discipline:
- A recipe is a VERSIONED, APPROVED document; `apply_recipe` refuses drafts.
- Failed sessions TARGET the recipe (which slice, which failure); they are
  never used as training labels — only sessions whose DETERMINISTIC outcome
  validates (task_satisfied, no violation) become demonstrations
  (labeling_policy = outcome_validated_demonstration_v1).
- Every apply writes a manifest (counts, drops, checks, hashes) and a
  lineage row — dataset → recipe → source store — so a later training run
  and evaluation can be traced back to the behavioral evidence.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from pydantic import BaseModel, Field

from shoprl.data.prompts_v2 import SYSTEM_PROMPT_CHAT_MIN
from shoprl.platform.milestones import _sessions
from shoprl.platform.store import PlatformStore

RECIPE_SCHEMA_VERSION = "recipe-v1"


class Recipe(BaseModel):
    recipe_id: str
    schema_version: str = RECIPE_SCHEMA_VERSION
    target_failure: str                    # e.g. PREMATURE_SEARCH
    finding: str = ""                      # human-readable evidence summary
    demonstration_source: dict = Field(default_factory=dict)
    #   {"model_version": ..., "require": {"task_satisfied": True, ...}}
    targeting_source: dict = Field(default_factory=dict)
    #   {"model_version": ..., "attribution": [...]} — informs slices only
    mixture: dict = Field(default_factory=dict)     # documented ratios
    deduplication: str = "scenario_id_first_v1"
    labeling_policy: str = "outcome_validated_demonstration_v1"
    evaluation_slices: list[str] = Field(default_factory=list)
    expected_effect: str = ""
    regression_risks: list[str] = Field(default_factory=list)
    approval_status: str = "draft"         # draft | approved | rejected
    approved_by: str = ""


def load_recipe(path: str | Path) -> Recipe:
    return Recipe.model_validate_json(Path(path).read_text())


def apply_recipe(store: PlatformStore, recipe: Recipe,
                 out_dir: str | Path) -> dict:
    if recipe.approval_status != "approved":
        raise PermissionError(
            f"recipe {recipe.recipe_id} is '{recipe.approval_status}' — "
            "a human must set approval_status='approved' before datasets "
            "are generated")
    sess = {s["session_id"]: s for s in _sessions(store)}

    def _match(s, cond):
        mv = cond.get("model_version")
        if mv and s["model_version"] != mv:
            return False
        for k, v in cond.get("require", {}).items():
            if bool((s["outcome"] or {}).get(k)) != v:
                return False
        return True

    demos, targets = [], []
    for s in sess.values():
        if _match(s, recipe.demonstration_source):
            demos.append(s)
        if _match(s, recipe.targeting_source):
            targets.append(s)

    # dedup: one demonstration per scenario (session ids embed scenario order)
    seen, kept, dropped_dup = set(), [], 0
    for s in sorted(demos, key=lambda x: x["session_id"]):
        key = s["session_id"].rsplit("-", 1)[-1]      # scenario index
        if key in seen:
            dropped_dup += 1
            continue
        seen.add(key)
        kept.append(s)

    sequences, dropped_invalid = [], 0
    for s in kept:
        o = s["outcome"] or {}
        if not o.get("task_satisfied") or o.get("safety_violation"):
            dropped_invalid += 1               # Rule 11: never unvalidated
            continue
        turns = store.db.execute(
            "SELECT agent, observation FROM turns WHERE session_id=?"
            " ORDER BY i", (s["session_id"],)).fetchall()
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_CHAT_MIN}]
        for j, (agent, obs) in enumerate(turns):
            if j == 0:
                msgs.append({"role": "user", "content": "(shopper opens)"})
            msgs.append({"role": "assistant", "content": agent})
            if obs:
                msgs.append({"role": "user", "content": obs})
        sequences.append({"session_id": s["session_id"], "messages": msgs})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{recipe.recipe_id}.jsonl"
    with data_path.open("w") as f:
        for row in sequences:
            f.write(json.dumps(row) + "\n")
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()[:16]
    manifest = {
        "recipe_id": recipe.recipe_id,
        "schema_version": recipe.schema_version,
        "created": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "dataset_sha256_16": digest,
        "sequences": len(sequences),
        "demonstration_candidates": len(demos),
        "dropped_duplicates": dropped_dup,
        "dropped_outcome_invalid": dropped_invalid,
        "targeting_sessions_informing_slices": len(targets),
        "mixture_documented": recipe.mixture,
        "labeling_policy": recipe.labeling_policy,
        "evaluation_slices": recipe.evaluation_slices,
        "quality_checks": {"violations_in_dataset": 0,
                           "all_outcome_validated": True},
    }
    (out_dir / f"{recipe.recipe_id}.manifest.json").write_text(
        json.dumps(manifest, indent=1))
    store.db.execute(
        "CREATE TABLE IF NOT EXISTS lineage(recipe_id TEXT, dataset_sha TEXT,"
        " created REAL, manifest TEXT)")
    store.db.execute("INSERT INTO lineage VALUES(?,?,?,?)",
                     (recipe.recipe_id, digest, time.time(),
                      json.dumps(manifest)))
    store.db.commit()
    return manifest
