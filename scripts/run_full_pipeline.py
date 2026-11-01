
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import itertools
import math
import time
from collections import Counter
from typing import Iterable

from models.dx11_full_pipeline import (
    BlendState,
    ConstantBuffer,
    DepthStencilState,
    DomainPoint,
    LaneSlot,
    Patch,
    ScissorRect,
    ShaderBinding,
    StreamRecord,
    ViewportRect,
    build_full_pipeline,
)
from pipeline.petri.colored import Binding, Fragment, Pixel, Primitive, RenderTarget, Vertex
from pipeline.petri.net import PetriNet
from pipeline.petri.snakes_bridge import snakes_to_state_space
from pipeline.petri.state_space import build_state_space
from pipeline.petri.to_snakes import translate as translate_to_snakes
from pipeline.verifier.dx11_ltl import verify_dx11_properties

def _lanes(net: PetriNet, stage: str) -> list[int]:
    return sorted(
        int(n.split("_")[2]) for n in net.places
        if n.startswith(f"{stage}_lane_") and n.endswith("_slot")
    )


def _full_pipeline_fire_all(net: PetriNet) -> Iterable[tuple[str, Binding]]:

    def tokens(place: str) -> list:
        p = net.places.get(place)
        return list(p.marking) if p is not None else []

    def _try(tname: str, b: Binding):
        if net.is_enabled(tname, b):
            yield (tname, b)
    layout_toks  = tokens("IA_input_layout")
    topo_toks    = tokens("IA_topology")
    vbuf_toks    = tokens("IA_vertex_buffer")
    for v, layout, topo in itertools.product(vbuf_toks, layout_toks, topo_toks):
        yield from _try("IA_fetch_vertex", {"v": v, "layout": layout, "topo": topo})

    vs_shader = tokens("VS_shader")
    vs_cbuf   = tokens("VS_cbuffer")
    for lane in _lanes(net, "VS"):
        for v, slot in itertools.product(tokens("VS_input"), tokens(f"VS_lane_{lane}_slot")):
            yield from _try(f"VS_dispatch_lane_{lane}", {"v": v, "slot": slot})

        for v, s, c in itertools.product(
            tokens(f"VS_lane_{lane}_input"), vs_shader, vs_cbuf
        ):
            x, y, z, w = v.clip_pos
            v_out = Vertex(
                vertex_id=v.vertex_id,
                clip_pos=(x * 0.5, y, z, w),
                instance_id=v.instance_id,
                attributes=v.attributes,
            )
            yield from _try(
                f"VS_exec_lane_{lane}",
                {"v": v, "cbuf": c, "vs_shader": s, "v_out": v_out},
            )

        for v in tokens(f"VS_lane_{lane}_output"):
            yield from _try(f"VS_collect_lane_{lane}", {"v": v, "slot": LaneSlot(lane=lane)})

    hs_bypass = tokens("HS_bypass")
    for v, bp in itertools.product(tokens("VS_output"), hs_bypass):
        yield from _try("HS_bypass_exec", {"v": v, "bypass": bp})

    gs_bypass = tokens("GS_bypass")
    for prim, bp in itertools.product(tokens("GS_input"), gs_bypass):
        yield from _try("GS_bypass_exec", {"prim": prim, "bypass": bp})

    so_bypass = tokens("SO_bypass")
    for prim, bp in itertools.product(tokens("SO_input"), so_bypass):
        yield from _try("SO_bypass_exec", {"prim": prim, "bypass": bp})

    dce = tokens("RS_depth_clip_enable")
    for prim in tokens("RS_input"):
        for d in dce:
            yield from _try("RS_clip_near", {"prim": prim, "dce": d})

    rs_state  = tokens("RS_rasterizer_state")
    cov_toks  = tokens("RS_coverage_mask")
    fill_toks = tokens("RS_fill_mode")
    laa_toks  = tokens("RS_line_aa_state")

    for prim in tokens("RS_clipped_primitive"):
        topo = prim.topology
        if topo == "triangle_list":
            for rs, cov, fill in itertools.product(rs_state, cov_toks, fill_toks):
                yield from _try(
                    "RS_rasterize_triangle",
                    {"prim": prim, "rs": rs, "cov": cov, "fill": fill},
                )
        elif topo == "line_list":
            for laa in laa_toks:
                yield from _try("RS_rasterize_line", {"prim": prim, "laa": laa})
        else:
            yield from _try("RS_rasterize_point", {"prim": prim})

    ps_shader = tokens("PS_shader")
    ps_cbuf   = tokens("PS_cbuffer")
    for lane in _lanes(net, "PS"):
        for frag, slot in itertools.product(tokens("PS_input"), tokens(f"PS_lane_{lane}_slot")):
            yield from _try(f"PS_dispatch_lane_{lane}", {"frag": frag, "slot": slot})

        for frag, s, c in itertools.product(
            tokens(f"PS_lane_{lane}_input"), ps_shader, ps_cbuf
        ):
            px = Pixel(
                pixel_x=frag.pixel_x,
                pixel_y=frag.pixel_y,
                color=(1.0, 0.0, 0.0, 1.0),
                depth=frag.depth,
                output_slot=0,
            )
            yield from _try(
                f"PS_exec_lane_{lane}",
                {"frag": frag, "cbuf": c, "ps_shader": s, "px": px},
            )

        for px in tokens(f"PS_lane_{lane}_output"):
            yield from _try(f"PS_collect_lane_{lane}", {"px": px, "slot": LaneSlot(lane=lane)})

    bs_toks  = tokens("OM_blend_state")
    dss_toks = tokens("OM_depth_stencil_state")
    dsv_toks = tokens("OM_dsv")
    ra_toks  = tokens("OM_rt_active")

    for px in tokens("PS_output"):
        slot = px.output_slot
        rt_toks = tokens(f"OM_rt_slot_{slot}")
        for rt, bs, ra in itertools.product(rt_toks, bs_toks, ra_toks):
            yield from _try(
                f"OM_write_rt_{slot}",
                {"px": px, "rt": rt, "bs": bs, "ra": ra},
            )
        for dss, rt in itertools.product(dss_toks, dsv_toks):
            yield from _try("OM_depth_test", {"px": px, "dss": dss, "rt": rt})

    for ra in tokens("OM_rt_in_use"):
        yield from _try("OM_release_rt", {"ra": ra})


# Transition names that _full_pipeline_fire_all enumerates ("happy path"
# bypass strategy). Used to scope the SNAKES engine's Transition.modes()
# exploration to the same progress-making subset, so both engines explore
# comparable reachability graphs and report comparable results.
_HAPPY_PATH_TRANSITIONS: tuple[str, ...] = (
    "IA_fetch_vertex",
    *(f"VS_dispatch_lane_{lane}" for lane in range(4)),
    *(f"VS_exec_lane_{lane}"     for lane in range(4)),
    *(f"VS_collect_lane_{lane}"  for lane in range(4)),
    "HS_bypass_exec",
    "GS_bypass_exec",
    "SO_bypass_exec",
    "RS_clip_near",
    "RS_rasterize_triangle",
    "RS_rasterize_line",
    "RS_rasterize_point",
    *(f"PS_dispatch_lane_{lane}" for lane in range(4)),
    *(f"PS_exec_lane_{lane}"     for lane in range(4)),
    *(f"PS_collect_lane_{lane}"  for lane in range(4)),
    "OM_write_rt_0",
    "OM_depth_test",
    "OM_release_rt",
)


def happy_path_for(net: PetriNet) -> list[str]:
    """The curated subset as it exists in *net* (35 at four lanes, 23 at two)."""
    return [t for t in _HAPPY_PATH_TRANSITIONS if t in net.transitions]


def firing_counts(ss) -> Counter:
    """Per-transition firing counts over every edge of the reachability graph."""
    return Counter(t for outs in ss.edges.values() for t, _ in outs)


def scc_stats(ss) -> dict:
    """Non-trivial SCCs (|SCC|>1), bottom SCCs, and self-loop edges by transition."""
    sccs = ss.sccs()
    comp = {m: i for i, c in enumerate(sccs) for m in c}
    bottom = [c for i, c in enumerate(sccs)
              if all(comp[d] == i for m in c for _, d in ss.successors(m))]
    selfloops = Counter(t for m, outs in ss.edges.items() for t, d in outs if d == m)
    return {
        "sccs": len(sccs),
        "non_trivial": sum(1 for c in sccs if len(c) > 1),
        "bottom": len(bottom),
        "selfloop_by_transition": dict(selfloops),
    }


def print_fire_counts(net: PetriNet, ss) -> None:
    subset = happy_path_for(net)
    fc = firing_counts(ss)
    never = [t for t in subset if fc.get(t, 0) == 0]
    print(f"\n  Per-transition firing counts ({len(subset)}-transition subset):")
    for t in subset:
        print(f"    {t:<24} {fc.get(t, 0):>9}")
    print(f"  Transitions that ever fire: {len(subset) - len(never)} / {len(subset)}")
    print(f"  Never fire: {never}")
    stages = ("IA", "VS", "HS", "Tess", "DS", "GS", "SO", "RS", "PS", "OM")
    print("  Per stage (subset / ever fire):  " + ",  ".join(
        f"{st} {sum(1 for t in subset if t.startswith(st + '_'))}/"
        f"{sum(1 for t in subset if t.startswith(st + '_') and fc.get(t, 0))}"
        for st in stages))


# At ~45 KB per state the limit is also a memory cap.  The default covers every
# four-lane run (largest: N = 4, 208,905 states, ~10 GB), so a larger N stops
# cleanly as INCONCLUSIVE instead of exhausting RAM.  The N = 5, 6 rows of
# Table III need LARGE_MAX_STATES (largest: one-lane N = 6, 2,006,893 states, ~90 GB).
DEFAULT_MAX_STATES = 300_000
LARGE_MAX_STATES = 3_000_000

# State invariants: a violating state is reachable even when BFS was cut off.
# Any other failure on a truncated space may come from the unexpanded frontier,
# whose states have no recorded successors and so look dead.
_STATE_INVARIANTS = frozenset({"P2", "P4"})


def status(name: str, satisfied: bool, truncated: bool) -> str:
    if satisfied:
        return "✓  PASS"
    if truncated and name not in _STATE_INVARIANTS:
        return "?  INCONCLUSIVE (truncated)"
    return "✗  FAIL"


def run(n_vertices: int, max_states: int = DEFAULT_MAX_STATES, engine: str = "custom",
        lanes: int = 4, fire_counts: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"  Full DX11 Pipeline  |  N = {n_vertices} vertices  |  lanes = {lanes}  |  engine = {engine}")
    print(f"{'='*60}")

    net = build_full_pipeline({"lanes": lanes})

    for i in range(n_vertices):
        v = Vertex(
            vertex_id=i,
            clip_pos=(float(i + 1), float(i), 0.5, 1.0),
            instance_id=0,
            attributes=(),
        )
        net.places["IA_vertex_buffer"].add_token(v)

    t0 = time.perf_counter()
    if engine == "snakes":
        snk_net, dropped = translate_to_snakes(net)
        if dropped:
            print(f"  Dropped (output-only vars, never fires): {len(dropped)} transitions")
        ss = snakes_to_state_space(
            snk_net, max_states=max_states, transitions=_HAPPY_PATH_TRANSITIONS
        )
    else:
        ss = build_state_space(net, _full_pipeline_fire_all, max_states=max_states)
    bfs_ms = (time.perf_counter() - t0) * 1000

    n_states = len(ss.states)
    es = ss.edge_stats()
    n_dead   = len(ss.deadlock_states())
    n_sdead  = len(ss.stutter_dead_states())
    sc = scc_stats(ss)

    print(f"  Reachable states  : {n_states}")
    print(f"  Transition firings: {es['firings']}")
    print(f"  Distinct (m,t,m') : {es['distinct_triples']}")
    print(f"  Distinct (m,m')   : {es['distinct_pairs']}  "
          f"(excluding self-loops: {es['distinct_pairs_no_selfloop']}; "
          f"self-loop firings: {es['selfloop_firings']})")
    print(f"  Deadlock states   : {n_dead} strict (no successor), "
          f"{n_sdead} stutter-dead (no successor or only self-loops)")
    print(f"  SCCs              : {sc['sccs']} total, {sc['non_trivial']} non-trivial, "
          f"{sc['bottom']} bottom; self-loops by transition {sc['selfloop_by_transition']}")
    print(f"  Truncated         : {ss.truncated}")
    print(f"  BFS time          : {bfs_ms:.2f} ms")

    t1 = time.perf_counter()
    results = verify_dx11_properties(ss)
    ltl_ms = (time.perf_counter() - t1) * 1000
    p3_strict = verify_dx11_properties_l3_strict(ss)

    print(f"\n  LTL results (check time {ltl_ms:.2f} ms):")
    for name, result in results.items():
        print(f"    {name}  {status(name, result.satisfied, ss.truncated)}  —  {result.formula}")
    print(f"    P3 under the strict (pre-round-3) deadlock notion: "
          f"{status('P3', p3_strict, ss.truncated)}")

    if fire_counts:
        print_fire_counts(net, ss)


def verify_dx11_properties_l3_strict(ss) -> bool:
    from pipeline.verifier.dx11_ltl import _build_p3, dead_states
    from pipeline.petri.ltl import LTLChecker
    return LTLChecker().check(ss, _build_p3(dead_states(ss, "strict"))).satisfied

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, default=None,
        help="Run with exactly N vertices (default: run 1–4)",
    )
    parser.add_argument(
        "--max-states", type=int, default=DEFAULT_MAX_STATES,
        help=f"BFS state-count limit (default: {DEFAULT_MAX_STATES})",
    )
    parser.add_argument(
        "--lanes", type=int, default=4,
        help="Number of 1-bounded VS/PS execution lanes (default: 4)",
    )
    parser.add_argument(
        "--engine", choices=("custom", "snakes"), default="custom",
        help="State-space engine: custom fire-all or SNAKES Transition.modes() (default: custom)",
    )
    parser.add_argument(
        "--fire-counts", action="store_true",
        help="Print per-transition firing counts over the curated subset",
    )
    args = parser.parse_args()

    targets = [args.n] if args.n is not None else [1, 2, 3, 4]
    for n in targets:
        run(n, max_states=args.max_states, engine=args.engine, lanes=args.lanes,
            fire_counts=args.fire_counts)

if __name__ == "__main__":
    main()
