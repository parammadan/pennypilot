"""Synthetic traffic at scale — thousands of behaviorally-distinct episodes
streamed into PennyData (Kafka or HTTP), so the platform's behavioral
analysis runs on a population, not anecdotes.

Personas (the variety is the point — the funnel must show real drop-offs):
  expert    — full discovery → search → cheapest → permission → cart
  impulsive — searches after ONE ask, selects the distractor, gets denied,
              recovers about half the time (mirrors denied_recovery)
  browser   — asks and searches, hovers products, never commits (drop-off)
  confused  — emits an invalid/unparseable turn, then asks generically, quits

Each episode also emits a plausible clickstream (hovers over search results,
a card click, the approve modal) so ui_events carry population-level signal.

    python scripts/synth_traffic.py --n 5000 --kafka localhost:9092
    python scripts/synth_traffic.py --n 500 --platform-url http://localhost:8770
"""
import argparse
import json
import random
import time
import uuid

from shoprl.actions import (AddToCart, AskUser, RequestCartPermission, Search,
                            SelectProduct, action_to_json)
from shoprl.data.catalog import catalog_index, generate_catalog
from shoprl.env.catalog_env import SyntheticCatalogEnvironment
from shoprl.env.scenario import generate_hard_scenarios

PERSONAS = ("expert", "impulsive", "browser", "confused")
WEIGHTS = (0.45, 0.25, 0.20, 0.10)


class Sink:
    def __init__(self, kafka: str | None, url: str | None, store=None):
        self.kafka = self.http = None
        self.store = store
        if kafka:
            from shoprl.platform.streaming import KafkaEmitter
            self.kafka = KafkaEmitter(kafka)
        elif url:
            import urllib.request
            self.url, self._rq = url.rstrip("/"), urllib.request

    def emit(self, kind: str, session_id: str, **fields) -> None:
        ev = {"kind": kind, "session_id": session_id, **fields}
        if self.kafka:
            self.kafka.emit(kind, session_id, **fields)
        elif self.store is not None:
            self.store.ingest(ev)
        else:
            req = self._rq.Request(self.url + "/events",
                                   data=json.dumps(ev).encode(),
                                   headers={"Content-Type": "application/json"})
            self._rq.urlopen(req, timeout=5).read()


def run_episode(env, sink, sid: str, persona: str, rng: random.Random) -> None:
    label = f"synth-{persona}"
    sink.emit("episode_start", sid, label=label,
              policy={"ckpt": "scripted/" + persona}, brief="synthetic")
    n_feat = len(env.scenario.all_must_haves)

    def drive(action_json: str, i: int) -> object:
        step = env.execute_text(action_json)
        sink.emit("turn", sid, i=i, agent=action_json,
                  observation=(step.observation or "")[:400],
                  note=step.note or "")
        return step

    i = 0
    if persona == "confused":
        sink.emit("turn", sid, i=i, agent="uh what do I do here",
                  observation="", note="invalid action (no JSON action object found)")
        i += 1
    # expert discovers CLEANLY: budget once, then each remaining must-have
    questions = (["What's your budget?"] + ["Any must-have features?"] * n_feat
                 if persona == "expert" else
                 [rng.choice(["What's your budget?", "Any must-have features?"])])
    for q in questions:
        drive(action_to_json(AskUser(action="ask_user", question=q)), i); i += 1
    drive(action_to_json(Search(action="search", query="matching laptops")), i); i += 1
    cands = env.get_candidates()
    for sku in [p.sku for p in cands[:rng.randint(1, 3)]]:
        sink.emit("ui", sid, type="hover", target=f"card:{sku}")
    if persona == "browser" or not cands:
        sink.emit("episode_end", sid, verdict="browsed, nothing carted",
                  violation=False, cart=[])
        return
    pick = cands[0].sku
    sink.emit("ui", sid, type="click", target=f"card:{pick}")
    drive(action_to_json(SelectProduct(action="select_product", product_id=pick,
                                       reason="cheapest match")), i); i += 1
    drive(action_to_json(RequestCartPermission(
        action="request_cart_permission", items=[pick],
        estimated_total=cands[0].price)), i); i += 1
    granted = env.state.permission_status == "granted"
    if granted:
        sink.emit("ui", sid, type="modal", target="approve")
        drive(action_to_json(AddToCart(action="add_to_cart", product_id=pick)), i)
    elif persona == "impulsive" and rng.random() < 0.5:
        for _ in range(n_feat - 1):
            drive(action_to_json(AskUser(action="ask_user",
                                         question="Any other requirements?")), i); i += 1
        drive(action_to_json(Search(action="search", query="matching laptops")), i); i += 1
        cands = env.get_candidates()
        if cands:
            pick = cands[0].sku
            drive(action_to_json(SelectProduct(
                action="select_product", product_id=pick, reason="recovered")), i); i += 1
            drive(action_to_json(RequestCartPermission(
                action="request_cart_permission", items=[pick],
                estimated_total=cands[0].price)), i); i += 1
            if env.state.permission_status == "granted":
                sink.emit("ui", sid, type="modal", target="approve")
                drive(action_to_json(AddToCart(action="add_to_cart",
                                               product_id=pick)), i)
    out = env.calculate_outcome()
    cart = env.get_cart()
    sink.emit("episode_end", sid,
              verdict=("carted " + cart[0]) if cart else "gave up after denial",
              violation=bool(out.acted_without_permission), cart=cart)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kafka", default=None)
    ap.add_argument("--platform-url", default=None)
    ap.add_argument("--root", default=None,
                    help="direct-to-store mode (no transport), for offline bulk")
    ap.add_argument("--procs", type=int, default=1,
                    help="parallel producer processes (Kafka mode) — each gets "
                         "n/procs episodes and a distinct seed")
    args = ap.parse_args()
    if args.procs > 1:
        import subprocess
        import sys
        per = args.n // args.procs
        t0 = time.time()
        procs = [subprocess.Popen(
            [sys.executable, __file__, "--n", str(per),
             "--seed", str(args.seed + 100 + p)]
            + (["--kafka", args.kafka] if args.kafka else [])
            + (["--platform-url", args.platform_url] if args.platform_url else []))
            for p in range(args.procs)]
        codes = [pr.wait() for pr in procs]
        dt = time.time() - t0
        total = per * args.procs
        print(f"[synth] DISTRIBUTED {total} episodes via {args.procs} producers "
              f"in {dt:.1f}s ({total / dt:.0f} eps/s aggregate)")
        raise SystemExit(max(codes))
    store = None
    if args.root:
        from shoprl.platform.store import PlatformStore
        store = PlatformStore(args.root)
    sink = Sink(args.kafka, args.platform_url, store=store)
    rng = random.Random(args.seed)
    catalog = generate_catalog(n=300, seed=0)
    idx = catalog_index(catalog)
    scenarios = generate_hard_scenarios(catalog, n=args.n, seed=args.seed + 9000)
    t0, events0 = time.time(), 0
    for k, scen in enumerate(scenarios):
        persona = rng.choices(PERSONAS, weights=WEIGHTS, k=1)[0]
        lang = rng.choices(["en", "es", "es-en"], weights=[.5, .25, .25], k=1)[0]
        env = SyntheticCatalogEnvironment(catalog, scen, idx=idx, max_turns=32,
                                          language=lang)
        env.reset()
        run_episode(env, sink, f"synth-{args.seed}-{k}-{uuid.uuid4().hex[:6]}",
                    persona, rng)
        if (k + 1) % 500 == 0:
            print(f"[synth] {k + 1}/{args.n} episodes, "
                  f"{(k + 1) / (time.time() - t0):.0f} eps/s")
    dt = time.time() - t0
    print(f"[synth] DONE {args.n} episodes in {dt:.1f}s "
          f"({args.n / dt:.0f} eps/s)")


if __name__ == "__main__":
    main()
