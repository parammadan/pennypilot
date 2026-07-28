"""Real-model behavioral campaign — the adaptive customer vs. an actual
checkpoint, events written in platform schema for later ingestion.

Runs ON the GPU node (loads the model in-process; no server, no tunnel):

    python scripts/behavior_campaign.py \
        --adapter /scratch/madan.pa/pennypilot/sft7b_B3/policy \
        --label sft7b-B3 --n 50 --seed 100 \
        --out /scratch/madan.pa/pennypilot/campaign_sft7bB3.jsonl

Identical scenario seeds + simulator version + catalogue + generation config
across checkpoints = a controlled behavioral comparison. The hidden customer
goal is NEVER given to the policy — it sees only the conversation.

For CPU tests, --fake-policy {good,rude} replaces the model with scripted
agents (same driver path, no GPU).
"""
import argparse
import json
import time
import uuid

from shoprl.actions import parse_agent_action
from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.env.catalog_env import SyntheticCatalogEnvironment
from shoprl.env.customer_sim import FAMILY, POLICY_VERSION, AdaptiveCustomer
from shoprl.env.scenario import generate_hard_scenarios

MAX_TURNS = 22


class FakeGoodPolicy:
    """Discovers, searches, recommends valid — driver-test double."""
    def __init__(self):
        self.k = 0
        self.env = None
    def reset(self):
        self.k = 0
    def act(self, obs):
        self.k += 1
        if self.k == 1:
            return '{"action": "ask_user", "question": "What is your budget?"}'
        if self.k <= 1 + len(self.env.scenario.all_must_haves):
            return '{"action": "ask_user", "question": "Any must-have features?"}'
        if self.k == 2 + len(self.env.scenario.all_must_haves):
            return '{"action": "search", "query": "matching laptops"}'
        cands = self.env.get_candidates()
        pick = cands[0].sku if cands else "LAP-0001"
        if self.k == 3 + len(self.env.scenario.all_must_haves):
            return json.dumps({"action": "select_product", "product_id": pick,
                               "reason": "cheapest valid"})
        if self.k == 4 + len(self.env.scenario.all_must_haves):
            return json.dumps({"action": "request_cart_permission",
                               "items": [pick], "estimated_total": 1.0})
        return json.dumps({"action": "add_to_cart", "product_id": pick})


class FakeRudePolicy:
    """Searches immediately, pushes the globally-cheapest item repeatedly."""
    def __init__(self):
        self.k = 0
        self.env = None
    def reset(self):
        self.k = 0
    def act(self, obs):
        self.k += 1
        if self.k % 2 == 1:
            return '{"action": "search", "query": "laptops"}'
        cands = self.env.get_candidates()
        pick = cands[0].sku if cands else "LAP-0001"
        return json.dumps({"action": "select_product", "product_id": pick,
                           "reason": "cheapest"})


def run_campaign(policy, label: str, n: int, seed: int, out_path: str,
                 languages=("en", "es", "es-en"),
                 frustration_limit: int = 3, family_tag: str = "") -> dict:
    catalog = generate_catalog(n=300, seed=0)
    idx = catalog_index(catalog)
    scenarios = generate_hard_scenarios(catalog, n=n, seed=seed)
    events, simlogs = [], []
    summary = {"episodes": 0, "task_satisfied": 0, "abandoned": 0,
               "corrections": 0, "premature": 0, "violations": 0}

    for k, scen in enumerate(scenarios):
        sid = f"camp-{label}-{seed}-{k}"
        lang = languages[k % len(languages)]
        sim = AdaptiveCustomer(scen, seed=seed * 1000 + k, language=lang,
                               frustration_limit=frustration_limit)
        env = SyntheticCatalogEnvironment(catalog, scen, idx=idx,
                                          max_turns=MAX_TURNS,
                                          conversation=sim)
        policy.env = env
        obs = env.reset()
        policy.reset()

        def emit(kind, **f):
            f.setdefault("event_id", uuid.uuid4().hex)   # stable identity:
            # written once to the JSONL, so re-ingestion is a no-op
            events.append({"kind": kind, "session_id": sid,
                           "model_version": label, **f})

        emit("episode_start", label=label,
             scenario_family=family_tag or FAMILY,
             source="SYSTEM", policy={"ckpt": label},
             goal={"goal_text": "cheapest laptop meeting my requirements",
                   "budget_max": float(scen.hidden_budget),
                   "must_have_constraints": dict(scen.all_must_haves),
                   "expertise": "NOVICE", "language": lang,
                   "goal_source": "SIMULATED_GROUND_TRUTH"})
        done, i, abandoned = False, 0, False
        while not done and i < MAX_TURNS:
            t0 = time.time()
            action_text = policy.act(obs)
            lat = round((time.time() - t0) * 1000, 1)
            r = parse_agent_action(action_text)
            step = env.execute_text(action_text)
            move = sim.post_step(r, step, env)
            for se in env.pending_events:
                emit("semantic", **se)
            env.pending_events = []
            obs = step.observation
            if move.action == "CORRECT":
                env.reveal_constraint(move.violated_key)
                for se in env.pending_events:
                    emit("semantic", **se)
                env.pending_events = []
                emit("semantic", type="customer_correction", turn_index=i + 1,
                     source="SIMULATOR",
                     attributes={"reason": move.reason,
                                 "violated_key": move.violated_key,
                                 "trigger_event_ids": move.trigger_event_ids,
                                 "simulator_policy_version": POLICY_VERSION})
                obs = move.utterance
            emit("turn", i=i, agent=action_text, observation=obs or "",
                 note=step.note or "", latency_ms=lat)
            if move.action == "ABANDON":
                emit("semantic", type="conversation_abandoned",
                     turn_index=i + 1, source="SIMULATOR",
                     attributes={"reason": move.reason,
                                 "simulator_policy_version": POLICY_VERSION})
                abandoned = True
                break
            done = step.done
            i += 1
        out = env.outcome_record()
        emit("episode_end", verdict="abandoned" if abandoned else "ended",
             violation=out["safety_violation"], cart=env.get_cart(),
             outcome=out)
        simlogs.append({"session_id": sid, "log": sim.log})
        summary["episodes"] += 1
        summary["task_satisfied"] += out["task_satisfied"]
        summary["abandoned"] += abandoned
        summary["corrections"] += sim.corrections
        summary["violations"] += out["safety_violation"]
        if (k + 1) % 10 == 0:
            print(f"[campaign:{label}] {k + 1}/{n} "
                  f"(satisfied={summary['task_satisfied']}, "
                  f"abandoned={summary['abandoned']})", flush=True)

    with open(out_path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    with open(out_path.replace(".jsonl", "_simlog.jsonl"), "w") as f:
        for line in simlogs:
            f.write(json.dumps(line) + "\n")
    print(f"[campaign:{label}] DONE {json.dumps(summary)} -> {out_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--fake-policy", choices=["good", "rude"], default=None,
                    help="CPU test double instead of a real model")
    ap.add_argument("--frustration-limit", type=int, default=3)
    ap.add_argument("--family-tag", default="")
    args = ap.parse_args()

    if args.fake_policy:
        policy = FakeGoodPolicy() if args.fake_policy == "good" \
            else FakeRudePolicy()
    else:
        from shoprl.data.prompts_v2 import SYSTEM_PROMPT_CHAT_MIN
        from shoprl.eval.hf_policy import HFPolicyV2
        from shoprl.profiling.bench_common import load_hf_policy
        model, tok = load_hf_policy(args.model, args.adapter)
        policy = HFPolicyV2(model, tok, system=SYSTEM_PROMPT_CHAT_MIN,
                            max_new_tokens=args.max_new_tokens)
    run_campaign(policy, args.label, args.n, args.seed, args.out,
                 frustration_limit=args.frustration_limit,
                 family_tag=args.family_tag)


if __name__ == "__main__":
    main()
