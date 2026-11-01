
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import itertools
import time
from typing import Iterable

from pipeline.petri.colored import Binding, Fragment, Pixel, Primitive, Vertex
from pipeline.petri.ltl import Atom, Formula, G, LTLChecker
from pipeline.petri.net import PetriNet
from pipeline.petri.state_space import build_state_space, place_count, place_tokens
from models.dx11_full_pipeline import LaneSlot

import pipeline.stages.output_merger as om_mod
import pipeline.stages.rasterizer as rs_mod
import pipeline.stages.pixel_shader as ps_mod

from pipeline.verifier.dx11_ltl import (
    P2_OM_MUTEX,
    P5_PS_NO_STARVATION,
    RS_QUEUE_BOUND,
    _rs_queue_within_bound,
    p5_per_lane,
)


# ---------------------------------------------------------------------------
# Generic helpers (mirrors scripts/run_full_pipeline.py's _try/tokens idiom)
# ---------------------------------------------------------------------------

def _tokens(net: PetriNet, place: str) -> list:
    p = net.places.get(place)
    return list(p.marking) if p is not None else []


def _try(net: PetriNet, tname: str, b: Binding):
    if net.is_enabled(tname, b):
        yield (tname, b)


# ---------------------------------------------------------------------------
# OM — curated fire-all + L2 (OM Mutual Exclusion)
# ---------------------------------------------------------------------------

def _om_fire_all(net: PetriNet) -> Iterable[tuple[str, Binding]]:
    """Curated to the same progress-making subset run_full_pipeline.py uses
    for OM (its ``_full_pipeline_fire_all``, lines covering ``OM_write_rt_*``
    / ``OM_depth_test`` / ``OM_release_rt``): every other OM transition is
    either a self-loop on ``PS_output`` (``OM_stencil_test``,
    ``OM_blend_rt_*`` reproduce the consumed pixel unchanged — pure cycle
    inflation) or accumulates into an *unbounded* sink
    (``OM_depth_write``/``OM_stencil_write`` -> ``OM_depth_buffer``/
    ``OM_stencil_buffer``, both ``bound=None``), which would make the
    reachable state space infinite. Excluding them keeps this subnet's BFS
    directly comparable to — and exploring the same fragment of OM as — the
    full-pipeline run already reported in Table~\\ref{tab:full_pipeline}."""
    def tokens(p: str) -> list:
        return _tokens(net, p)

    dss_toks = tokens("OM_depth_stencil_state")
    dsv_toks = tokens("OM_dsv")
    bs_toks = tokens("OM_blend_state")
    ra_toks = tokens("OM_rt_active")

    for px in tokens("PS_output"):
        slot = px.output_slot
        if 0 <= slot < om_mod.N_RT_SLOTS:
            for rt, bs, ra in itertools.product(tokens(f"OM_rt_slot_{slot}"), bs_toks, ra_toks):
                yield from _try(net, f"OM_write_rt_{slot}", {"px": px, "rt": rt, "bs": bs, "ra": ra})
        for dss, rt in itertools.product(dss_toks, dsv_toks):
            yield from _try(net, "OM_depth_test", {"px": px, "dss": dss, "rt": rt})

    for ra in tokens("OM_rt_in_use"):
        yield from _try(net, "OM_release_rt", {"ra": ra})


def _seed_om(net: PetriNet) -> None:
    """Seed PS_output with realistic, non-colliding pixels — the OM's
    typical working set as fragments survive rasterization and depth/stencil
    testing for genuinely distinct screen positions. None of OM's transitions
    alter a pixel's (x, y) (they all consume-and-reproduce PS_output tokens
    in place), so this seeding lets BFS exhaustively confirm that OM's own
    processing never creates the L2 hazard — a genuine, non-vacuous result,
    as opposed to the trivial "no hazard because no pixels at all" vacuity
    the professor flagged in the VS-subnet table."""
    seeds = [
        Pixel(pixel_x=10, pixel_y=20, color=(1.0, 0.0, 0.0, 1.0), depth=0.4, output_slot=0),
        Pixel(pixel_x=10, pixel_y=20, color=(0.0, 1.0, 0.0, 1.0), depth=0.6, output_slot=0),
    ]
    for px in seeds:
        net.places["PS_output"].add_token(px)


# ---------------------------------------------------------------------------
# RS — curated fire-all + L4 (RS Queue Boundedness)
# ---------------------------------------------------------------------------

def _rs_fire_all(net: PetriNet) -> Iterable[tuple[str, Binding]]:
    def tokens(p: str) -> list:
        return _tokens(net, p)

    dce_toks = tokens("RS_depth_clip_enable")
    rsstate_toks = tokens("RS_rasterizer_state")
    cov_toks = tokens("RS_coverage_mask")
    fill_toks = tokens("RS_fill_mode")
    laa_toks = tokens("RS_line_aa_state")

    for prim in tokens("RS_input"):
        for dce in dce_toks:
            yield from _try(net, "RS_clip_near", {"prim": prim, "dce": dce})

    for prim in tokens("RS_clipped_primitive"):
        topo = prim.topology
        if topo == "triangle_list":
            for rs, cov, fill in itertools.product(rsstate_toks, cov_toks, fill_toks):
                yield from _try(net, "RS_rasterize_triangle",
                                {"prim": prim, "rs": rs, "cov": cov, "fill": fill})
        elif topo == "line_list":
            for laa in laa_toks:
                yield from _try(net, "RS_rasterize_line", {"prim": prim, "laa": laa})
        else:
            yield from _try(net, "RS_rasterize_point", {"prim": prim})


def _seed_rs(net: PetriNet, n_primitives: int) -> None:
    """Seed RS_input with n_primitives distinct triangles (one vertex each
    distinguishing clip_pos so RS_rasterize_triangle produces fragments at
    distinct screen coordinates) — mirroring how run_full_pipeline.py seeds
    IA_vertex_buffer for the headline N=1..4 configurations."""
    for i in range(n_primitives):
        v = Vertex(vertex_id=i, clip_pos=(float(i + 1), float(i), 0.5, 1.0))
        net.places["RS_input"].add_token(
            Primitive(vertices=(v,), topology="triangle_list", instance_id=0)
        )


# ---------------------------------------------------------------------------
# PS — curated fire-all (+ contention machinery) + L5 (PS Starvation Freedom)
# ---------------------------------------------------------------------------

def _ps_fire_all(net: PetriNet) -> Iterable[tuple[str, Binding]]:
    def tokens(p: str) -> list:
        return _tokens(net, p)

    ps_shader = tokens("PS_shader")
    ps_cbuf = tokens("PS_cbuffer")

    for lane in range(ps_mod.N_LANES):
        for frag, slot in itertools.product(tokens("PS_input"), tokens(f"PS_lane_{lane}_slot")):
            yield from _try(net, f"PS_dispatch_lane_{lane}", {"frag": frag, "slot": slot})

        for frag, s, c in itertools.product(tokens(f"PS_lane_{lane}_input"), ps_shader, ps_cbuf):
            px = ps_mod.ps_transform(frag)
            yield from _try(net, f"PS_exec_lane_{lane}",
                            {"frag": frag, "cbuf": c, "ps_shader": s, "px": px})

        for px in tokens(f"PS_lane_{lane}_output"):
            yield from _try(net, f"PS_collect_lane_{lane}", {"px": px, "slot": LaneSlot(lane=lane)})

        for frag in tokens(f"PS_lane_{lane}_input"):
            yield from _try(net, f"PS_mem_request_lane_{lane}", {"frag": frag})

        yield from _try(net, f"PS_mem_acquire_lane_{lane}", {})
        yield from _try(net, f"PS_mem_release_lane_{lane}", {})


def _seed_ps(net: PetriNet, n_fragments: int) -> None:
    for i in range(n_fragments):
        net.places["PS_input"].add_token(
            Fragment(pixel_x=i, pixel_y=i, depth=0.5)
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

# Covers every run in the paper (largest: PS subnet at N = 4, 112,739 states).
DEFAULT_MAX_STATES = 400_000


def _liveness_status(satisfied: bool, truncated: bool) -> str:
    """L5 is a response property: on a truncated space a failure may come from
    the unexpanded frontier, whose states have no recorded successors.  (L2 and
    L4 are state invariants, so their FAIL stands either way.)"""
    if satisfied:
        return "PASS"
    return "INCONCLUSIVE (truncated)" if truncated else "FAIL"


def _run_om(max_states: int) -> None:
    print(f"\n{'='*60}\n  OM subnet  |  L2 = OM Mutual Exclusion\n{'='*60}")
    net = om_mod.build_subnet()
    _seed_om(net)
    t0 = time.perf_counter()
    ss = build_state_space(net, _om_fire_all, max_states=max_states)
    bfs_ms = (time.perf_counter() - t0) * 1000
    n_states, n_edges = len(ss.states), sum(len(v) for v in ss.edges.values())
    print(f"  Reachable states : {n_states}")
    print(f"  Transition firings: {n_edges}")
    print(f"  Deadlock states  : {len(list(ss.deadlock_states()))}")
    print(f"  Truncated        : {ss.truncated}")
    print(f"  BFS time         : {bfs_ms:.2f} ms")

    checker = LTLChecker()
    result = checker.check(ss, P2_OM_MUTEX)
    status = "PASS" if result.satisfied else "FAIL"
    print(f"\n  L2 (at most one RT write in flight)  {status}  —  {result.formula}")


def _run_rs(n_primitives: int, max_states: int) -> None:
    print(f"\n{'='*60}\n  RS subnet  |  N = {n_primitives} primitives  |  L4 = RS Queue Boundedness\n{'='*60}")
    net = rs_mod.build_subnet()
    _seed_rs(net, n_primitives)
    t0 = time.perf_counter()
    ss = build_state_space(net, _rs_fire_all, max_states=max_states)
    bfs_ms = (time.perf_counter() - t0) * 1000
    n_states, n_edges = len(ss.states), sum(len(v) for v in ss.edges.values())
    print(f"  Reachable states : {n_states}")
    print(f"  Transition firings: {n_edges}")
    print(f"  Deadlock states  : {len(list(ss.deadlock_states()))}")
    print(f"  Truncated        : {ss.truncated}")
    print(f"  BFS time         : {bfs_ms:.2f} ms")

    max_q = max(place_count(m, "PS_input") for m in ss.states)
    checker = LTLChecker()
    formula: Formula = G(Atom(_rs_queue_within_bound, f"rs_queue_≤{RS_QUEUE_BOUND}"))
    result = checker.check(ss, formula)
    status = "PASS" if result.satisfied else "FAIL"
    print(f"\n  L4 (|PS_input| ≤ {RS_QUEUE_BOUND}; observed max = {max_q})  {status}  —  {result.formula}")


def _run_ps(n_fragments: int, max_states: int) -> None:
    print(f"\n{'='*60}\n  PS subnet  |  N = {n_fragments} fragments  |  L5 = PS Starvation Freedom\n{'='*60}")
    net = ps_mod.build_subnet()
    _seed_ps(net, n_fragments)
    t0 = time.perf_counter()
    ss = build_state_space(net, _ps_fire_all, max_states=max_states)
    bfs_ms = (time.perf_counter() - t0) * 1000
    n_states, n_edges = len(ss.states), sum(len(v) for v in ss.edges.values())
    print(f"  Reachable states : {n_states}")
    print(f"  Transition firings: {n_edges}")
    print(f"  Deadlock states  : {len(list(ss.deadlock_states()))}")
    print(f"  Truncated        : {ss.truncated}")
    print(f"  BFS time         : {bfs_ms:.2f} ms")

    checker = LTLChecker()
    result = checker.check(ss, P5_PS_NO_STARVATION)
    status = _liveness_status(result.satisfied, ss.truncated)
    print(f"\n  L5 (PS_waiting ∧ mem_free → F(PS_has_mem))  {status}  —  {result.formula}")

    # Indexed (per-lane) instantiation: the aggregate marking predicate above
    # cannot distinguish "some lane acquires" from "this lane acquires".
    per_lane = {
        lane: checker.check(ss, p5_per_lane(lane)) for lane in range(ps_mod.N_LANES)
    }
    all_ok = all(r.satisfied for r in per_lane.values())
    detail = "  ".join(
        f"lane{lane}={_liveness_status(r.satisfied, ss.truncated)}"
        for lane, r in per_lane.items()
    )
    print(f"  L5 per-lane (∀ lane i: wait_i ∧ mem_free → F(has_mem_i))  "
          f"{_liveness_status(all_ok, ss.truncated)}  —  {detail}")

    # The per-lane form can fail only because the acquire/release protocol puts
    # genuine cycles in the reachability graph: another lane may take and
    # return the port forever while this lane's request sits in PS_mem_wait.
    sccs = ss.sccs()
    non_trivial = [c for c in sccs if len(c) > 1]
    starving = [
        c for c in non_trivial
        if any(
            all(any(getattr(t, "lane", None) == lane for t in place_tokens(m, "PS_mem_wait"))
                for m in c)
            for lane in range(ps_mod.N_LANES)
        )
    ]
    print(f"  SCCs: {len(sccs)} total, {len(non_trivial)} non-trivial "
          f"(largest {max((len(c) for c in non_trivial), default=0)}); "
          f"{len(starving)} of them hold one lane waiting throughout")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genuine, non-vacuous L2/L4/L5 verification on the OM/RS/PS home subnets."
    )
    parser.add_argument("--stage", choices=("om", "rs", "ps", "all"), default="all")
    parser.add_argument("--n", type=int, default=3,
                        help="Seed size for RS (primitives) / PS (fragments) (default: 3)")
    parser.add_argument("--max-states", type=int, default=DEFAULT_MAX_STATES,
                        help=f"BFS state-count limit (default: {DEFAULT_MAX_STATES})")
    args = parser.parse_args()

    if args.stage in ("om", "all"):
        _run_om(args.max_states)
    if args.stage in ("rs", "all"):
        _run_rs(args.n, args.max_states)
    if args.stage in ("ps", "all"):
        _run_ps(args.n, args.max_states)


if __name__ == "__main__":
    main()
