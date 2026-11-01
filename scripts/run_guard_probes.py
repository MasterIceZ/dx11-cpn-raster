"""Probe what the VS and PS output guards do with out-of-range values on the
composed net (round-3 review, Q5).

The guards are enabling conditions, so a token that fails one is not routed
anywhere: the transition simply never fires for it.  This script makes that
concrete by exploring the happy-path restriction with one deliberately invalid
value injected, and reports where the token ends up and what each checker says.

  * PS: the rasterizer emits fragments with depth outside [0, 1].
  * VS: a seeded vertex has SV_Position w = 0 or w = NaN.

Usage:
    uv run python scripts/run_guard_probes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math
from typing import Callable

from models.dx11_full_pipeline import build_full_pipeline
from pipeline.petri.colored import Fragment, Vertex
from pipeline.petri.net import Multiset, PetriNet
from pipeline.petri.state_space import build_state_space, place_count
from pipeline.verifier.dx11_ltl import _PIPELINE_BUFFERS

from run_full_pipeline import _full_pipeline_fire_all
from run_mutation_tests import _all_props, PROPS


def _set_fragment_depth(depth: float) -> Callable[[PetriNet], None]:
    def mutate(net: PetriNet) -> None:
        for a in net.output_arcs(net.transitions["RS_rasterize_point"]):
            if a.place.name == "PS_input":
                a.expression = lambda _b, _d=depth: Multiset.of(
                    Fragment(pixel_x=0, pixel_y=0, depth=_d))
    return mutate


def _probe(label: str, n: int, mutate: Callable[[PetriNet], None] | None = None,
           w: float = 1.0) -> None:
    net = build_full_pipeline()
    for i in range(n):
        net.places["IA_vertex_buffer"].add_token(
            Vertex(vertex_id=i, clip_pos=(float(i + 1), float(i), 0.5, w),
                   instance_id=0, attributes=()))
    if mutate is not None:
        mutate(net)
    ss = build_state_space(net, _full_pipeline_fire_all, max_states=50_000)
    props = _all_props(ss)
    print(f"\n  {label}  (N={n})")
    print(f"    states={len(ss.states)}  strict-dead={len(ss.deadlock_states())}  "
          f"stutter-dead={len(ss.stutter_dead_states())}")
    for m in ss.stutter_dead_states():
        flow = {p: c for p, _t, c in sorted(m, key=lambda e: e[0])
                if ("_lane_" in p and not p.endswith("_slot")) or p in _PIPELINE_BUFFERS
                or p.startswith("OM_written") or p.endswith("discarded_fragment")
                or p.endswith("discarded_pixel")}
        held = [p for p in net.places if p.endswith("_slot") and "_lane_" in p
                and place_count(m, p) == 0]
        listed = [p for p in _PIPELINE_BUFFERS if place_count(m, p)]
        print(f"    terminal marking: flow tokens {flow}; lane semaphores held {held}; "
              f"L3-listed buffers non-empty {listed}")
    print("    checkers: " + "  ".join(
        f"{p}={'.' if props[p] else 'X'}" for p in PROPS)
        + f"  L3strict={'.' if props['L3strict'] else 'X'}")


def main() -> None:
    print("=" * 72)
    print("  Guard probes on the composed net (happy-path restriction)")
    print("  . = holds, X = violated; L3 = stutter notion")
    print("=" * 72)
    _probe("control: valid depth 0.5, w = 1", 1)
    _probe("PS guard: fragment depth 1.5", 1, _set_fragment_depth(1.5))
    _probe("PS guard: fragment depth -0.5", 1, _set_fragment_depth(-0.5))
    _probe("PS guard: fragment depth 1.5", 2, _set_fragment_depth(1.5))
    _probe("VS guard: SV_Position w = 0", 1, w=0.0)
    _probe("VS guard: SV_Position w = NaN", 1, w=math.nan)


if __name__ == "__main__":
    main()
