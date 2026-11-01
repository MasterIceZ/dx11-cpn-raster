
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import run_full_pipeline` resolves

import argparse
from dataclasses import dataclass
from typing import Callable, Iterable

from pipeline.petri.net import Multiset, PetriNet
from pipeline.petri.ltl import Atom, G, LTLChecker
from pipeline.petri.state_space import build_state_space
from pipeline.petri.colored import Vertex
from pipeline.verifier.dx11_ltl import (
    verify_dx11_properties,
    P5_PS_NO_STARVATION,
    _build_p3,
    dead_states,
)
from models.dx11_full_pipeline import build_full_pipeline

import pipeline.stages.pixel_shader as ps_mod

from run_full_pipeline import _full_pipeline_fire_all
from run_om_rs_ps_subnets import _ps_fire_all


# ---------------------------------------------------------------------------
# Fault-injection helpers
# ---------------------------------------------------------------------------

def _disable(net: PetriNet, *names_or_prefixes: str) -> int:
    """Disable transitions by name or name-prefix by forcing their guard false.

    Keeping the Transition object present (rather than deleting it) means the
    curated fire-all enumerators' ``is_enabled`` lookups still resolve; the
    transition simply never becomes enabled, modelling a broken / missing
    arc in the pipeline."""
    n = 0
    for tname, t in net.transitions.items():
        if any(tname == p or tname.startswith(p) for p in names_or_prefixes):
            t.guard = lambda _b: False
            n += 1
    return n


def _seed_vertices(net: PetriNet, n: int, *, collide: bool = False) -> None:
    for i in range(n):
        # collide=True forces identical clip_pos -> identical fragment/pixel
        # coordinates so two pixels race for the same (x, y) at the OM.
        cp = (1.0, 0.0, 0.5, 1.0) if collide else (float(i + 1), float(i), 0.5, 1.0)
        net.places["IA_vertex_buffer"].add_token(
            Vertex(vertex_id=i, clip_pos=cp, instance_id=0, attributes=())
        )


# ---------------------------------------------------------------------------
# Fault catalogue: each entry mutates a net and names the property it should
# break.  ``run`` returns the satisfied-flag of *every* property, so the report
# shows not only that the targeted checker is sensitive to its fault but that
# the other five are unaffected by it -- i.e. checker specificity, not just
# sensitivity.
# ---------------------------------------------------------------------------

PROPS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5", "L6")

_PID_TO_KEY = {p: f"P{p[1:]}" for p in PROPS}


@dataclass
class Mutation:
    pid: str            # targeted property, e.g. "L1"
    fault: str          # human description of the injected fault
    run: Callable[[int], dict[str, bool]]   # pid -> satisfied flag
    # Properties whose atomic propositions are actually interpretable on the
    # net this fault is injected into.  Anything outside this set is reported
    # as "-" rather than as a pass or a violation.
    scope: tuple[str, ...] = PROPS


def _all_props(ss, **kw) -> dict[str, bool]:
    """Outcome of all six checkers (L3 under the stutter deadlock notion), plus
    "L3strict": L3 under the pre-round-3 notion (dead = no successor at all),
    so the report shows exactly which cells the notion change moves."""
    res = verify_dx11_properties(ss, **kw)
    out = {pid: res[key].satisfied for pid, key in _PID_TO_KEY.items()}
    out["L3strict"] = LTLChecker().check(ss, _build_p3(dead_states(ss, "strict"))).satisfied
    return out


def _full(n: int, mutate: Callable[[PetriNet], None], *, collide: bool = False):
    net = build_full_pipeline()
    _seed_vertices(net, n, collide=collide)
    mutate(net)
    ss = build_state_space(net, _full_pipeline_fire_all, max_states=50_000)
    return net, ss


def _m_l1(n: int) -> dict[str, bool]:
    # Fault: drop every render-target write sink, so a fetched vertex can
    # never reach a written pixel nor a sanctioned discard.
    _, ss = _full(n, lambda net: _disable(net, "OM_write_rt_", "OM_depth_write",
                                          "OM_stencil_write"))
    return _all_props(ss)


def _m_l1_drop(n: int) -> dict[str, bool]:
    # Silently lose vertex 0 at IA fetch: the output arc to VS_input yields
    # nothing for it.  The old "some pixel eventually written" L1 passes this;
    # token conservation must not.
    def lossy(net: PetriNet) -> None:
        t = net.transitions["IA_fetch_vertex"]
        for a in net.output_arcs(t):
            if a.place.name == "VS_input":
                a.expression = lambda b: (Multiset() if b["v"].vertex_id == 0
                                          else Multiset.of(b["v"]))
    _, ss = _full(n, lossy)
    return _all_props(ss)


def _m_l2(n: int) -> dict[str, bool]:
    # Duplicate the render-target context token (and lift the 1-bounds that
    # would otherwise silently block the second write) so two OM writes can be
    # in flight at once.  N>=2 pixels all land on (0,0), so this is the
    # overlapping-fragment case the OM must serialize.
    def dup_context(net: PetriNet) -> None:
        for pname in ("OM_rt_active", "OM_rt_in_use"):
            net.places[pname].bound = None
        ra = next(iter(net.places["OM_rt_active"].marking))
        net.places["OM_rt_active"].add_token(ra)
    _, ss = _full(n, dup_context)
    return _all_props(ss)


def _m_l3(n: int) -> dict[str, bool]:
    # Fault: break rasterization so primitives strand in an inter-stage buffer;
    # the pipeline then deadlocks with a buffer still non-empty.
    _, ss = _full(n, lambda net: _disable(net, "RS_rasterize_"))
    return _all_props(ss)


def _m_l4(n: int) -> dict[str, bool]:
    # Fault: declare an unrealistically tight queue bound (0) the rasterizer
    # output immediately violates.
    _, ss = _full(n, lambda net: None)
    return _all_props(ss, rs_queue_bound=0)


def _double_rasterizer_output(net: PetriNet) -> None:
    """Net-level amplification fault: every rasterize transition deposits two
    copies of its fragment in PS_input instead of one."""
    for tname in ("RS_rasterize_triangle", "RS_rasterize_line", "RS_rasterize_point"):
        for a in net.output_arcs(net.transitions[tname]):
            if a.place.name == "PS_input":
                orig = a.expression
                a.expression = lambda b, _f=orig: Multiset(
                    {tok: 2 * c for tok, c in _f(b).items()})


def _m_l4_fanout(n: int) -> dict[str, bool]:
    # Net-level fault: RS emits two fragments per primitive.  The unmutated
    # rasterizer yields exactly one fragment per vertex, so max |PS_input| = N
    # and N is the tight queue bound for this net (see the control row); the
    # fault must push the queue past it.
    _, ss = _full(n, _double_rasterizer_output)
    return _all_props(ss, rs_queue_bound=n)


def _m_l4_control(n: int) -> dict[str, bool]:
    # Control for the row above: same bound N, no fault.  Must hold everywhere.
    _, ss = _full(n, lambda net: None)
    return _all_props(ss, rs_queue_bound=n)


def _m_l5(n: int) -> dict[str, bool]:
    # Fault: break the memory-port acquire so a waiting PS lane can never
    # obtain the free port -> starvation.
    net = ps_mod.build_subnet()
    for i in range(n):
        net.places["PS_input"].add_token(ps_mod.Fragment(pixel_x=i, pixel_y=i, depth=0.5))
    _disable(net, "PS_mem_acquire_lane_")
    ss = build_state_space(net, _ps_fire_all, max_states=50_000)
    # L5 is the only property whose places live in the PS subnet; the other
    # five reference composed-net places that this subnet does not contain, so
    # they are reported (vacuously satisfied) for completeness of the matrix.
    props = _all_props(ss)
    props["L5"] = LTLChecker().check(ss, P5_PS_NO_STARVATION).satisfied
    return props


def _m_l6(n: int) -> dict[str, bool]:
    # Fault: drop the render-target release so the OM_rt_active flag, once
    # consumed by a write, is never restored -> not perpetually available.
    _, ss = _full(n, lambda net: _disable(net, "OM_release_rt"))
    return _all_props(ss)


MUTATIONS: list[Mutation] = [
    Mutation("L1", "Remove render-target write sinks (no pixel ever written)", _m_l1),
    Mutation("L1", "Silently drop one vertex at IA fetch (lossy arc)", _m_l1_drop),
    Mutation("L2", "Duplicate RT context token (two writes in flight)", _m_l2),
    Mutation("L3", "Break rasterization (primitive stranded in buffer)", _m_l3),
    Mutation("L4", "Declare queue bound = 0 (immediately exceeded)", _m_l4),
    Mutation("L4", "RS emits 2 fragments/primitive (bound = N)", _m_l4_fanout),
    Mutation("L5", "Break PS memory-port acquire (waiting lane starves)", _m_l5,
             scope=("L5",)),
    Mutation("L6", "Drop OM_release_rt (rt_active never restored)", _m_l6),
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fault-injection: confirm each LTL checker discriminates "
                    "(targeted property must flip to FAIL under its fault)."
    )
    ap.add_argument("--n", type=int, default=2, help="seed size (default: 2)")
    args = ap.parse_args()

    print(f"\n{'='*88}")
    print(f"  DX11 CPN — Fault-Injection Specificity Matrix  (N={args.n})")
    print(f"  Cell = outcome of each checker under the injected fault "
          f"(. = holds, X = violated)")
    print(f"{'='*88}")
    header = "  ".join(f"{p:>2}" for p in PROPS) + "  L3strict"
    print(f"  {'Target':<7} {'Injected fault':<50} {header}")
    print(f"  {'-'*7} {'-'*50} {'-'*len(header)}")

    sensitive = True
    specific = True
    for m in MUTATIONS:
        outcome = m.run(args.n)
        cells = []
        for p in PROPS:
            if p not in m.scope:
                cells.append(f"{'-':>2}")
                continue
            ok = outcome[p]
            cells.append(f"{('.' if ok else 'X'):>2}")
            if p == m.pid and ok:
                sensitive = False          # targeted checker failed to flip
            if p != m.pid and not ok:
                specific = False           # collateral violation
        if "L3" in m.scope:
            cells.append(f"{('.' if outcome['L3strict'] else 'X'):>8}")
        else:
            cells.append(f"{'-':>8}")
        print(f"  {m.pid:<7} {m.fault:<50} {'  '.join(cells)}", flush=True)

    print(f"  {'-'*7} {'-'*50} {'-'*len(header)}")
    ctl = _m_l4_control(args.n)
    print(f"  {'control':<7} {'No fault, queue bound = N':<50} "
          + "  ".join(f"{('.' if ctl[p] else 'X'):>2}" for p in PROPS)
          + f"  {('.' if ctl['L3strict'] else 'X'):>8}")
    print("  (L3 = stutter deadlock notion; L3strict = dead only when nothing is enabled)")
    if sensitive:
        print("  Sensitivity: every targeted property is violated by its own fault.")
    else:
        print("  WARNING: a targeted checker did NOT flip — it may be vacuously passing.")
    if specific:
        print("  Specificity: no fault violates a property other than its target.")
    else:
        print("  NOTE: at least one fault violates a non-targeted property "
              "(see the matrix above).")
    print()
    sys.exit(0 if sensitive else 1)


if __name__ == "__main__":
    main()
