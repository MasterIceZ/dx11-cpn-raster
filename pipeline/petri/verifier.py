
from __future__ import annotations

import argparse
import importlib
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pipeline.petri.colored import Vertex
from pipeline.petri.ltl import (
    And, Atom, CheckResult, F, G, Implies, LTLChecker, Not, Or, Formula,
)
from pipeline.petri.net import Binding, PetriNet
from pipeline.petri.state_space import (
    FrozenMarking, StateSpace, build_state_space, place_count,
)





def _make_vs_fire_all(net: PetriNet):
    from pipeline.stages.vertex_shader import ShaderBinding

    def fire_all(n: PetriNet) -> Iterable[tuple[str, Binding]]:
        shader_toks = list(n.places["VS_shader"].marking)
        cbuf_toks   = list(n.places["VS_cbuffer"].marking)
        input_toks  = list(n.places["VS_input"].marking)
        for s, c, v in itertools.product(shader_toks, cbuf_toks, input_toks):


            x, y, z, w = v.clip_pos
            v_out = Vertex(
                vertex_id=v.vertex_id,
                clip_pos=(x * 0.5, y, z, w),
                instance_id=v.instance_id,
                attributes=v.attributes,
            )
            b: Binding = {"v": v, "cbuf": c, "vs_shader": s, "v_out": v_out}
            if n.is_enabled("VS_exec", b):
                yield ("VS_exec", b)

    return fire_all


_FIRE_ALL_REGISTRY: dict[str, Any] = {
    "vertex_shader": _make_vs_fire_all,
}





def _check_safety(ss: StateSpace, net: PetriNet) -> list[CheckResult]:
    checker = LTLChecker()
    results: list[CheckResult] = []
    for name, place in net.places.items():
        if place.bound is None:
            continue
        bound = place.bound
        formula: Formula = G(
            Atom(
                lambda m, p=name, b=bound: place_count(m, p) <= b,
                f"|{name}| ≤ {bound}",
            )
        )
        results.append(checker.check(ss, formula))
    return results

def _check_liveness(ss: StateSpace, net: PetriNet) -> list[CheckResult]:
    checker = LTLChecker()
    results: list[CheckResult] = []

    _RESOURCE_KEYWORDS = ("cbuffer", "shader", "sampler", "srv", "uav", "mem")
    _FLOW_KEYWORDS     = ("input", "queue")

    flow_places = [
        n for n in net.places
        if any(kw in n.lower() for kw in _FLOW_KEYWORDS)
        and not any(rk in n.lower() for rk in _RESOURCE_KEYWORDS)
    ]
    for name in flow_places:
        formula: Formula = G(
            Implies(
                Atom(lambda m, p=name: place_count(m, p) > 0, f"{name}_nonempty"),
                F(Atom(lambda m, p=name: place_count(m, p) == 0, f"{name}_empty")),
            )
        )
        results.append(checker.check(ss, formula))
    return results

def _check_bounded(ss: StateSpace, net: PetriNet) -> list[CheckResult]:

    checker = LTLChecker()
    results: list[CheckResult] = []
    for name, place in net.places.items():
        if place.bound is None:
            continue
        bound = place.bound
        lo_formula: Formula = G(
            Atom(lambda m, p=name: place_count(m, p) >= 0, f"|{name}| ≥ 0")
        )
        hi_formula: Formula = G(
            Atom(
                lambda m, p=name, b=bound: place_count(m, p) <= b,
                f"|{name}| ≤ {bound}",
            )
        )
        results.append(checker.check(ss, lo_formula))
        results.append(checker.check(ss, hi_formula))
    return results





_STAGE_ORDER = ["IA", "VS", "HS", "Tess", "DS", "GS", "SO", "RS", "PS", "OM"]


_STAGE_COLORS = [
    "#fff0f0", "#fff8e8", "#fffff0", "#f0fff0",
    "#e8ffff", "#f0f0ff", "#f8e8ff", "#ffe8f8",
    "#ffe8e8", "#e8f0ff",
]


_STAGE_PLACE_COLORS = [
    "#ffcccc", "#ffe0aa", "#ffff99", "#aaffaa",
    "#aaffff", "#aaaaff", "#ddaaff", "#ffaadd",
    "#ffaaaa", "#aaccff",
]

def _stage_index(name: str) -> int:
    for i, s in enumerate(_STAGE_ORDER):
        if name.upper().startswith(s.upper() + "_") or name.upper() == s.upper():
            return i
    return len(_STAGE_ORDER) - 1

def _hierarchical_pos(
    place_nodes: list[str],
    trans_nodes: list[str],
    x_gap: float = 3.0,
    y_gap: float = 1.0,
) -> dict[str, tuple[float, float]]:
    from collections import defaultdict


    stage_places: dict[int, list[str]] = defaultdict(list)
    stage_trans:  dict[int, list[str]] = defaultdict(list)

    for n in place_nodes:
        stage_places[_stage_index(n)].append(n)
    for n in trans_nodes:
        stage_trans[_stage_index(n)].append(n)

    pos: dict[str, tuple[float, float]] = {}
    n_stages = len(_STAGE_ORDER)

    for si in range(n_stages):
        x_base = si * x_gap

        places = sorted(stage_places[si])
        trans  = sorted(stage_trans[si])


        for j, n in enumerate(places):
            y = -(j - (len(places) - 1) / 2.0) * y_gap
            pos[n] = (x_base, y)


        for j, n in enumerate(trans):
            y = -(j - (len(trans) - 1) / 2.0) * y_gap
            pos[n] = (x_base + x_gap * 0.45, y)

    return pos

def render_net(net: PetriNet, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    G_nx = nx.DiGraph()

    place_nodes: list[str] = []
    trans_nodes: list[str] = []

    for name, place in net.places.items():
        count = place.token_count()
        bound_s = f" k={place.bound}" if place.bound is not None else ""
        tok_s   = f" [{count}]" if count else ""

        short = name
        for s in _STAGE_ORDER:
            pfx = s + "_"
            if name.startswith(pfx):
                short = name[len(pfx):]
                break
        label = f"{short}\n{place.color_type.__name__}{bound_s}{tok_s}"
        G_nx.add_node(name, label=label, kind="place", stage=_stage_index(name))
        place_nodes.append(name)

    for name in net.transitions:
        short = name
        for s in _STAGE_ORDER:
            pfx = s + "_"
            if name.startswith(pfx):
                short = name[len(pfx):]
                break
        G_nx.add_node(name, label=short, kind="trans", stage=_stage_index(name))
        trans_nodes.append(name)

    for arc in net.arcs:
        if arc.direction == "in":
            G_nx.add_edge(arc.place.name, arc.transition.name)
        else:
            G_nx.add_edge(arc.transition.name, arc.place.name)

    n_nodes = len(place_nodes) + len(trans_nodes)
    large = n_nodes > 60

    if large:
        pos = _hierarchical_pos(place_nodes, trans_nodes, x_gap=2.5, y_gap=0.75)
        figw, figh = 55, 28
        node_size_p = 550
        node_size_t = 450
        font_p = 5
        font_t = 5
        edge_alpha = 0.18
        edge_width = 0.7
        arrow_size = 8
    else:
        pos = nx.spring_layout(G_nx, seed=42, k=2.5)
        figw, figh = 14, 9
        node_size_p = 2800
        node_size_t = 2200
        font_p = 7
        font_t = 8
        edge_alpha = 0.8
        edge_width = 1.5
        arrow_size = 18

    fig, ax = plt.subplots(figsize=(figw, figh))
    ax.set_title(f"CPN: {net.name}  ({len(place_nodes)} places, {len(trans_nodes)} transitions)",
                 fontsize=12 if large else 14, fontweight="bold")
    ax.axis("off")


    if large and pos:
        xs = [p[0] for p in pos.values()]
        x_min, x_max = min(xs), max(xs)

        from collections import defaultdict
        stage_xs: dict[int, list[float]] = defaultdict(list)
        for n, (x, _y) in pos.items():
            stage_xs[_stage_index(n)].append(x)
        for si, sxs in stage_xs.items():
            lo = min(sxs) - 0.6
            hi = max(sxs) + 0.6
            ax.axvspan(lo, hi, alpha=0.25,
                       color=_STAGE_COLORS[si % len(_STAGE_COLORS)], zorder=0)

            ax.text((lo + hi) / 2, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 1,
                    _STAGE_ORDER[si], ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#333333", zorder=5)


    place_colors = [
        _STAGE_PLACE_COLORS[_stage_index(n) % len(_STAGE_PLACE_COLORS)]
        for n in place_nodes
    ]

    nx.draw_networkx_nodes(G_nx, pos, nodelist=place_nodes, ax=ax,
                           node_shape="o", node_size=node_size_p,
                           node_color=place_colors, linewidths=0.8,
                           edgecolors="#336699")
    nx.draw_networkx_nodes(G_nx, pos, nodelist=trans_nodes, ax=ax,
                           node_shape="s", node_size=node_size_t,
                           node_color="#ffffaa", linewidths=0.8,
                           edgecolors="#997700")
    nx.draw_networkx_labels(G_nx, pos, labels={n: G_nx.nodes[n]["label"] for n in place_nodes},
                            ax=ax, font_size=font_p, font_color="#003366")
    nx.draw_networkx_labels(G_nx, pos, labels={n: G_nx.nodes[n]["label"] for n in trans_nodes},
                            ax=ax, font_size=font_t, font_color="#664400")
    nx.draw_networkx_edges(G_nx, pos, ax=ax, arrows=True,
                           arrowstyle="-|>", arrowsize=arrow_size,
                           edge_color="#555555", alpha=edge_alpha, width=edge_width,
                           connectionstyle="arc3,rad=0.05",
                           min_source_margin=12, min_target_margin=12)


    if large:
        ys = [p[1] for p in pos.values()]
        y_top = max(ys) + 1.2
        from collections import defaultdict as _dd
        stage_xs2: dict[int, list[float]] = _dd(list)
        for n, (x, _y) in pos.items():
            stage_xs2[_stage_index(n)].append(x)
        for si, sxs in stage_xs2.items():
            cx = (min(sxs) + max(sxs)) / 2
            ax.text(cx, y_top, _STAGE_ORDER[si],
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#333333",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
                    zorder=6)

    legend = [
        mpatches.Patch(color="#aaddff", label="Place (circle)"),
        mpatches.Patch(color="#ffffaa", label="Transition (square)"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    dpi = 180 if large else 150
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path





@dataclass
class VerificationReport:
    stage: str
    num_states: int
    num_deadlocks: int
    truncated: bool
    checks: dict[str, list[CheckResult]]
    render_path: Path | None

    def print_summary(self) -> None:
        w = 64
        print("=" * w)
        print(f"  Stage: {self.stage}")
        print(f"  States: {self.num_states}  |  Deadlocks: {self.num_deadlocks}"
              + ("  [TRUNCATED]" if self.truncated else ""))
        print("=" * w)

        all_pass = True
        for check_name, results in self.checks.items():
            sat  = [r for r in results if r.satisfied]
            unsat = [r for r in results if not r.satisfied]
            status = "PASS" if not unsat else "FAIL"
            if unsat:
                all_pass = False
            print(f"\n  [{status}] {check_name.upper()}"
                  f"  ({len(sat)}/{len(results)} formulas satisfied)")
            for r in unsat:
                cx_len = len(r.counterexample) if r.counterexample else 0
                print(f"    ✗  {r.formula}")
                if r.counterexample:
                    print(f"       counterexample path length: {cx_len}")
                if r.note:
                    print(f"       note: {r.note}")

        print()
        print("  " + ("ALL CHECKS PASSED ✓" if all_pass else "VIOLATIONS FOUND ✗"))
        if self.render_path:
            print(f"  Net diagram → {self.render_path}")
        print("=" * w)

def verify_stage(
    stage: str,
    *,
    checks: set[str] | None = None,
    config: dict | None = None,
    max_states: int = 5_000,
    render_to: Path | None = None,
) -> VerificationReport:
    checks = checks or {"safety", "liveness", "bounded"}


    module_path = Path("pipeline/stages") / f"{stage}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Stage module not found: {module_path}")
    mod = importlib.import_module(f"pipeline.stages.{stage}")

    net: PetriNet = mod.build_subnet(config)


    print(f"\nStage : {stage}")
    print(f"Net   : {net.name}")
    print(f"Places: {len(net.places)}")
    for name, place in net.places.items():
        bound_s = f"  k={place.bound}" if place.bound is not None else ""
        tok_s   = f"  [{place.token_count()} tok]" if place.token_count() else ""
        print(f"         {name}  ({place.color_type.__name__}){bound_s}{tok_s}")
    print(f"Transitions: {len(net.transitions)}")
    for name in net.transitions:
        print(f"         {name}")
    print(f"Arcs  : {len(net.arcs)}")
    color_types = {p.color_type.__name__ for p in net.places.values()}
    print(f"Color sets: {', '.join(sorted(color_types))}")


    from pipeline.stages.vertex_shader import ConstantBuffer
    cbuf = ConstantBuffer(slot=0, size=16)
    net.places["VS_cbuffer"].add_token(cbuf)
    for i in range(3):
        v = Vertex(vertex_id=i, clip_pos=(float(i + 1), float(i), 0.5, 1.0))
        net.places["VS_input"].add_token(v)


    fire_all_factory = _FIRE_ALL_REGISTRY.get(stage)
    if fire_all_factory is None:
        raise NotImplementedError(
            f"No fire_all factory registered for stage '{stage}'. "
            "Add one to _FIRE_ALL_REGISTRY in pipeline/petri/verifier.py."
        )
    fire_all = fire_all_factory(net)

    print(f"\nBuilding state space (max {max_states} states)…")
    ss = build_state_space(net, fire_all, max_states=max_states)
    print(f"Done: {len(ss.states)} states, {len(ss.deadlock_states())} deadlocks"
          + (" [TRUNCATED]" if ss.truncated else ""))


    check_results: dict[str, list[CheckResult]] = {}
    if "safety"  in checks:
        check_results["safety"]  = _check_safety(ss, net)
    if "liveness" in checks:
        check_results["liveness"] = _check_liveness(ss, net)
    if "bounded"  in checks:
        check_results["bounded"]  = _check_bounded(ss, net)


    rendered: Path | None = None
    if render_to is not None:
        try:
            rendered = render_net(net, render_to)
            print(f"Net rendered → {rendered}")
        except Exception as exc:
            print(f"Render failed: {exc}")

    report = VerificationReport(
        stage=stage,
        num_states=len(ss.states),
        num_deadlocks=len(ss.deadlock_states()),
        truncated=ss.truncated,
        checks=check_results,
        render_path=rendered,
    )
    report.print_summary()
    return report





def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Formal verification of a single DX11 pipeline stage CPN subnet."
    )
    parser.add_argument(
        "--net",
        required=True,
        help="Path to stage module, e.g. pipeline/stages/vertex_shader.py",
    )
    parser.add_argument(
        "--check",
        default="safety,liveness,bounded",
        help="Comma-separated list of checks to run (default: safety,liveness,bounded)",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=5_000,
        help="BFS state-space exploration cap (default: 5000)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the rendered net PNG (e.g. output/vs_net.png)",
    )
    args = parser.parse_args()

    stage = Path(args.net).stem
    checks = {c.strip() for c in args.check.split(",")}
    render_to = Path(args.output) if args.output else None

    verify_stage(
        stage,
        checks=checks,
        max_states=args.max_states,
        render_to=render_to,
    )

if __name__ == "__main__":
    _cli()
