"""Structural checks on the composed DX11 CPN.

Three machine-checked facts that the paper reports as claims:

1.  Per-stage place / transition attribution of the composed net, alongside the
    number of *progress-making* transitions each stage contributes to the
    curated fire-all subset (Section III-E).

2.  Net-zero verification of the excluded transitions: an excluded transition is
    net-zero iff its input and output arc multisets are identical, in which case
    it can never create a new marking and its exclusion cannot hide any
    reachable state.  Arc expressions are closures over a binding, so they are
    evaluated against a probe binding that maps every variable name to a unique
    sentinel; a transition whose expressions construct fresh tokens from bound
    ones raises on the sentinel and is reported as *not* statically net-zero.

3.  Triviality of the strongly connected components of the reachability graph
    for each explored configuration: every SCC must be a singleton, and the only
    self-loop edges must come from OM_depth_test.  This is the empirical form of
    the acyclicity premise that licenses the AF fixpoint in Section III-F.

Usage:
    uv run python scripts/run_structural_checks.py            # 1 and 2 only
    uv run python scripts/run_structural_checks.py --scc 1 2 3
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
from collections import Counter

from models.dx11_full_pipeline import build_full_pipeline
from pipeline.petri.net import Multiset, PetriNet
from pipeline.petri.state_space import build_state_space, place_count
from pipeline.petri.colored import Vertex

from run_full_pipeline import _HAPPY_PATH_TRANSITIONS, _full_pipeline_fire_all

STAGES = ["IA", "VS", "HS", "Tess", "DS", "GS", "SO", "RS", "PS", "OM"]


# ---------------------------------------------------------------------------
# 1.  Per-stage attribution
# ---------------------------------------------------------------------------

def _stage_of(name: str) -> str | None:
    for s in STAGES:
        if name.startswith(s + "_"):
            return s
    return None


def report_attribution(net: PetriNet) -> None:
    places = Counter()
    trans = Counter()
    progress = Counter()
    unattributed_p: list[str] = []
    unattributed_t: list[str] = []

    for name in net.places:
        s = _stage_of(name)
        if s is None:
            unattributed_p.append(name)
        else:
            places[s] += 1
    for name in net.transitions:
        s = _stage_of(name)
        if s is None:
            unattributed_t.append(name)
            continue
        trans[s] += 1
        if name in _HAPPY_PATH_TRANSITIONS:
            progress[s] += 1

    print("\n[1] Per-stage attribution of the composed net")
    print(f"    {'Stage':<8} {'Places':>7} {'Trans':>7} {'Progress':>9}")
    for s in STAGES:
        print(f"    {s:<8} {places[s]:>7} {trans[s]:>7} {progress[s]:>9}")
    print(f"    {'TOTAL':<8} {sum(places.values()):>7} {sum(trans.values()):>7} "
          f"{sum(progress.values()):>9}")
    print(f"    net.places={len(net.places)}  net.transitions={len(net.transitions)}  "
          f"happy-path={len(_HAPPY_PATH_TRANSITIONS)}")
    if unattributed_p:
        print(f"    unattributed places     : {unattributed_p}")
    if unattributed_t:
        print(f"    unattributed transitions: {unattributed_t}")

    # Interface places: a place whose name-prefix stage differs from the stage
    # of at least one transition it is arc-connected to.  These are the places
    # on which composition-by-name-matching joins two stages.
    iface: set[str] = set()
    for arc in net.arcs:
        ps, ts = _stage_of(arc.place.name), _stage_of(arc.transition.name)
        if ps is not None and ts is not None and ps != ts:
            iface.add(f"{arc.place.name} ({ps}<->{ts})")
    print(f"    inter-stage interface places ({len(iface)}):")
    for n in sorted(iface):
        print(f"        {n}")


# ---------------------------------------------------------------------------
# 2.  Net-zero verification of excluded transitions
# ---------------------------------------------------------------------------

class _Sentinel:
    """Opaque, hashable stand-in for a bound variable's value."""

    __slots__ = ("var",)

    def __init__(self, var: str) -> None:
        self.var = var

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Sentinel) and other.var == self.var

    def __hash__(self) -> int:
        return hash(("_Sentinel", self.var))

    def __repr__(self) -> str:
        return f"<{self.var}>"


class _ProbeBinding(dict):
    """Binding that yields a fresh sentinel for any variable name asked for."""

    def __missing__(self, key: str) -> _Sentinel:
        s = _Sentinel(key)
        self[key] = s
        return s


def _arc_profile(net: PetriNet, tname: str, direction: str):
    """Multiset of (place, token) the transition consumes / produces, or None
    if the arc expressions cannot be evaluated symbolically."""
    t = net.transitions[tname]
    arcs = net.input_arcs(t) if direction == "in" else net.output_arcs(t)
    profile: Counter = Counter()
    for arc in arcs:
        b = _ProbeBinding()
        try:
            produced = arc.evaluate(b)
        except Exception:
            return None
        for token, count in produced.items():
            profile[(arc.place.name, token)] += count
    return profile


def report_net_zero(net: PetriNet) -> None:
    excluded = [t for t in net.transitions if t not in _HAPPY_PATH_TRANSITIONS]
    net_zero: list[str] = []
    not_net_zero: list[str] = []
    unresolved: list[str] = []

    for tname in excluded:
        pre = _arc_profile(net, tname, "in")
        post = _arc_profile(net, tname, "out")
        if pre is None or post is None:
            unresolved.append(tname)
        elif pre == post:
            net_zero.append(tname)
        else:
            not_net_zero.append(tname)

    # Unbounded token producers: the output arc multiset strictly contains the
    # input arc multiset, so every firing adds tokens and the transition can
    # fire forever.  Excluding them is what keeps the state space finite.
    generators: list[str] = []
    for tname in excluded:
        pre, post = _arc_profile(net, tname, "in"), _arc_profile(net, tname, "out")
        if pre is None or post is None or pre == post:
            continue
        if all(post[k] >= v for k, v in pre.items()) and sum(post.values()) > sum(pre.values()):
            generators.append(tname)

    print(f"\n[2] Net-zero verification of the {len(excluded)} excluded transitions")
    print(f"    net-zero (identical input/output arc multisets): {len(net_zero)}")
    print(f"    not net-zero (genuine data paths)              : {len(not_net_zero)}")
    print(f"    token-constructing, not statically net-zero    : {len(unresolved)}")
    print(f"    unbounded token producers (subset of 'not net-zero'): {len(generators)}"
          f"  {sorted(generators)}")
    print(f"    net-zero set: {sorted(net_zero)}")
    if unresolved:
        print(f"    unresolved  : {sorted(unresolved)}")

    # Also confirm the one retained net-zero transition inside the happy path.
    retained = [
        t for t in _HAPPY_PATH_TRANSITIONS
        if t in net.transitions
        and (_arc_profile(net, t, "in") is not None)
        and _arc_profile(net, t, "in") == _arc_profile(net, t, "out")
    ]
    print(f"    net-zero transitions retained in the happy path: {retained}")


# ---------------------------------------------------------------------------
# 3.  SCC triviality of the reachability graph
# ---------------------------------------------------------------------------

def report_scc(n_vertices: int, lanes: int = 4, max_states: int = 400_000) -> None:
    net = build_full_pipeline({"lanes": lanes})
    for i in range(n_vertices):
        net.places["IA_vertex_buffer"].add_token(
            Vertex(vertex_id=i, clip_pos=(float(i + 1), float(i), 0.5, 1.0),
                   instance_id=0, attributes=())
        )
    ss = build_state_space(net, _full_pipeline_fire_all, max_states=max_states)

    sccs = ss.sccs()
    non_trivial = [c for c in sccs if len(c) > 1]
    self_loops: Counter = Counter()
    for src, outs in ss.edges.items():
        for tname, dst in outs:
            if dst == src:
                self_loops[tname] += 1

    # Bottom SCCs: components with no edge leaving the component.  L6's verdict
    # is a statement about these, so it matters whether the terminal marking is
    # the only one.
    comp_of = {}
    for i, c in enumerate(sccs):
        for m in c:
            comp_of[m] = i
    bottom = []
    for i, c in enumerate(sccs):
        if all(comp_of[d] == i for m in c for _, d in ss.successors(m)):
            bottom.append(c)
    dead = ss.deadlock_states()
    bottom_dead = [c for c in bottom if len(c) == 1 and next(iter(c)) in dead]

    max_queue = max(place_count(m, "PS_input") for m in ss.states)

    tag = f"N={n_vertices}" + (f" ({lanes}L)" if lanes != 4 else "")
    print(f"\n[3] SCC triviality  {tag}")
    print(f"    states={len(ss.states)}  edges={sum(len(v) for v in ss.edges.values())}"
          f"  truncated={ss.truncated}")
    print(f"    SCCs={len(sccs)}  non-trivial (|SCC|>1)={len(non_trivial)}")
    print(f"    bottom SCCs={len(bottom)}  of which terminal markings={len(bottom_dead)}")
    print(f"    max |PS_input| (RS fragment queue) = {max_queue}")
    print(f"    self-loop edges by transition: {dict(self_loops)}")
    trivial = len(non_trivial) == 0 and set(self_loops) <= {"OM_depth_test"}
    print(f"    verdict: {'ALL SCCs TRIVIAL (mod OM_depth_test stutter)' if trivial else 'NON-TRIVIAL SCC PRESENT'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scc", type=int, nargs="*", default=None,
                    help="vertex counts to run the SCC check for")
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--max-states", type=int, default=400_000)
    args = ap.parse_args()

    net = build_full_pipeline()
    report_attribution(net)
    report_net_zero(net)

    for n in (args.scc or []):
        report_scc(n, lanes=args.lanes, max_states=args.max_states)


if __name__ == "__main__":
    main()
