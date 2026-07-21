"""Record a demo trajectory (CPU): oracle policy, Spanglish hard scenario.

    python scripts/make_demo_transcript.py --out runs/demo/oracle_esen.json

Writes the bundle scripts/demo_browser.py replays. Honest labeling: the bundle
carries `policy` so the demo can say whether it's an oracle or a trained-model
trajectory (swap in model transcripts once captured on the cluster).
"""
from __future__ import annotations

import argparse
import json
import os

from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.env.catalog_env import SyntheticCatalogEnvironment
from shoprl.env.scenario import generate_hard_scenarios
from shoprl.eval.v2_policies import OracleGoodV2
from shoprl.transcript import transcript_record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog-n", type=int, default=150)
    ap.add_argument("--catalog-seed", type=int, default=0)
    ap.add_argument("--scenario-seed", type=int, default=11)
    ap.add_argument("--scenario-index", type=int, default=0)
    ap.add_argument("--language", default="es-en", choices=["en", "es", "es-en"])
    ap.add_argument("--out", default="runs/demo/oracle_esen.json")
    args = ap.parse_args()

    catalog = generate_catalog(n=args.catalog_n, seed=args.catalog_seed)
    idx = catalog_index(catalog)
    scen = generate_hard_scenarios(catalog, n=args.scenario_index + 1,
                                   seed=args.scenario_seed)[args.scenario_index]
    env = SyntheticCatalogEnvironment(catalog, scen, idx=idx,
                                      language=args.language)
    policy = OracleGoodV2()
    env.reset()
    policy.reset(scen, idx)
    done = False
    while not done:
        done = env.execute_text(policy.act()).done

    record = transcript_record(env, env.calculate_outcome())
    record["opener"] = env.opener
    bundle = {"catalog_n": args.catalog_n, "catalog_seed": args.catalog_seed,
              "language": args.language, "policy": "oracle (scripted ceiling)",
              "record": record}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"[demo] {scen.scenario_id} ({args.language}) reward "
          f"{record['reward']['total']:+.3f} -> {args.out}")


if __name__ == "__main__":
    main()
