"""Explore the composed DX11 CPN with SNAKES over *all* transitions (not the
curated happy-path subset), minus the five input-less generator transitions
whose unbounded token production would make the state space infinite.

Reports state-space size, terminal markings, per-transition firing coverage,
and the six LTL outcomes -- i.e. what the unrestricted model looks like
compared with the happy-path restriction reported in Table IV.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import time
from collections import Counter

from models.dx11_full_pipeline import build_full_pipeline
from pipeline.petri.colored import Vertex
from pipeline.petri.snakes_bridge import snakes_to_state_space
from pipeline.petri.to_snakes import translate as translate_to_snakes
from pipeline.petri.ltl import LTLChecker
from pipeline.petri.state_space import place_count
from pipeline.verifier.dx11_ltl import (
    verify_dx11_properties, _written_count, _captured_count, _discarded_count,
    _build_p3, dead_states, _any_buffer_nonempty, _PIPELINE_BUFFERS,
)
from run_full_pipeline import _HAPPY_PATH_TRANSITIONS

GENERATORS: tuple[str, ...] = (
    "IA_emit_draw_cmd", "HS_emit_patch_constants", "DS_emit_clip_distance",
    "OM_depth_write", "OM_stencil_write",
)


def run(n: int, max_states: int) -> None:
    net = build_full_pipeline()
    for i in range(n):
        net.places["IA_vertex_buffer"].add_token(
            Vertex(vertex_id=i, clip_pos=(float(i + 1), float(i), 0.5, 1.0), instance_id=0, attributes=()))
    snk, dropped = translate_to_snakes(net)
    names = [t.name for t in snk.transition() if t.name not in GENERATORS]
    print(f"\n{'='*64}\n  All-transition SNAKES exploration | N = {n} | {len(names)} transitions "
          f"(excluding {len(GENERATORS)} generators; {len(dropped)} untranslatable dropped)\n{'='*64}")
    t0 = time.perf_counter()
    ss = snakes_to_state_space(snk, max_states=max_states, transitions=names)
    print(f"  Reachable states  : {len(ss.states)}   (happy path: see Table IV)")
    print(f"  Edges             : {sum(len(v) for v in ss.edges.values())}")
    print(f"  Terminal markings : {len(ss.deadlock_states())} strict (nothing enabled), "
          f"{len(ss.stutter_dead_states())} stutter-dead (only self-loops enabled)")
    es = ss.edge_stats()
    print(f"  Distinct (m,m')   : {es['distinct_pairs']}  (non-self-loop {es['distinct_pairs_no_selfloop']}; "
          f"self-loop firings {es['selfloop_firings']})")
    print(f"  Truncated         : {ss.truncated}")
    print(f"  BFS time          : {time.perf_counter() - t0:.1f} s")
    fired = Counter(t for edges in ss.edges.values() for t, _ in edges)
    extra = sorted(t for t in fired if t not in _HAPPY_PATH_TRANSITIONS)
    print(f"  Transitions fired : {len(fired)} / {len(names)}; outside happy path: {len(extra)}")
    print("    " + ", ".join(extra))
    # L1 diagnosis: is the conservation predicate ever satisfied at all, and if
    # so, is the failure a lost token or an unfair cycle that never reaches it?
    acc = [_written_count(m) + _captured_count(m) + _discarded_count(m)
           for m in ss.states]
    n_sat = sum(1 for a in acc if a == n)
    print(f"  L1 accounting     : max accounted = {max(acc)} / {n}; "
          f"{n_sat} of {len(ss.states)} states fully accounted")

    report_sccs(ss, n)

    res = verify_dx11_properties(ss)
    for k, r in res.items():
        print(f"    {k}  {'PASS' if r.satisfied else 'FAIL'}  {r.formula}  {r.note}")
    p3s = LTLChecker().check(ss, _build_p3(dead_states(ss, "strict")))
    print(f"    P3 under the strict (pre-round-3) deadlock notion: {'PASS' if p3s.satisfied else 'FAIL'}")
    if not res["P3"].satisfied:
        cx = res["P3"].counterexample or []
        last = cx[-1] if cx else None
        if last is not None:
            bufs = {p: place_count(last, p) for p in _PIPELINE_BUFFERS if place_count(last, p)}
            print(f"    P3 counterexample: length {len(cx)}, dead state buffers {bufs}, "
                  f"enabled self-loops {sorted({t for t, _ in ss.successors(last)})}")
        sd_bad = [m for m in ss.stutter_dead_states() if _any_buffer_nonempty(m)]
        print(f"    stutter-dead states with a non-empty buffer: {len(sd_bad)}")
        print(f"      non-empty buffers there: "
              f"{dict(Counter(p for m in sd_bad for p in _PIPELINE_BUFFERS if place_count(m, p)))}")


def report_sccs(ss, n: int) -> None:
    """Q2: what cycles does the graph contain?  Non-trivial SCCs (|SCC|>1),
    transitions on their internal edges, self-loops, and bottom SCCs."""
    sccs = ss.sccs()
    comp = {m: i for i, c in enumerate(sccs) for m in c}
    nontriv = [i for i, c in enumerate(sccs) if len(c) > 1]
    internal = Counter()
    for m, outs in ss.edges.items():
        for t, d in outs:
            if d != m and comp[d] == comp[m]:
                internal[t] += 1
    selfloops = Counter(t for m, outs in ss.edges.items() for t, d in outs if d == m)
    bottom = [i for i, c in enumerate(sccs)
              if all(comp[d] == i for m in c for _, d in ss.successors(m))]
    print(f"  SCCs              : {len(sccs)} total; non-trivial (|SCC|>1): {len(nontriv)} "
          f"sizes {sorted((len(sccs[i]) for i in nontriv), reverse=True)}")
    print(f"  Edges inside non-trivial SCCs by transition: {dict(internal)}")
    print(f"  Self-loop firings by transition ({len(selfloops)} transitions): {dict(selfloops)}")
    acc = lambda m: _written_count(m) + _captured_count(m) + _discarded_count(m)
    print(f"  Bottom SCCs       : {len(bottom)}; sizes {sorted(len(sccs[i]) for i in bottom)}")
    for i in bottom:
        c = sccs[i]
        print(f"    bottom SCC size {len(c)}: accounted counts {sorted({acc(m) for m in c})}, "
              f"buffers non-empty {any(_any_buffer_nonempty(m) for m in c)}, "
              f"stutter-dead {all(ss.is_stutter_dead(m) for m in c)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=None, help="vertices (default: 1 and 2)")
    ap.add_argument("--max-states", type=int, default=100_000)
    args = ap.parse_args()
    for n in ([args.n] if args.n is not None else [1, 2]):
        run(n, args.max_states)


if __name__ == "__main__":
    main()
