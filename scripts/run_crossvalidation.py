"""Cross-validate custom BFS engine against SNAKES for the full DX11 pipeline.

Runs both engines for N=1..3 (and optionally N=4 with --include-n4) and
compares state counts, deadlock counts, and LTL outcomes side by side.
Reproduces the cross-validation reported in the paper (Section IV-G).

Usage
-----
    uv run python scripts/run_crossvalidation.py              # N=1,2,3
    uv run python scripts/run_crossvalidation.py --include-n4 # N=1,2,3,4 (~2.5h)
    uv run python scripts/run_crossvalidation.py --n 2        # single N
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dx11_full_pipeline import (
    LaneSlot,
    build_full_pipeline,
)
from pipeline.petri.colored import Binding, Fragment, Pixel, Primitive, RenderTarget, Vertex
from pipeline.petri.net import PetriNet
from pipeline.petri.snakes_bridge import snakes_to_state_space
from pipeline.petri.state_space import FrozenMarking, StateSpace, build_state_space
from pipeline.petri.to_snakes import translate as translate_to_snakes
from pipeline.verifier.dx11_ltl import _build_p3, dead_states, verify_dx11_properties
from pipeline.petri.ltl import LTLChecker

# ---------------------------------------------------------------------------
# Re-use the same fire-all and happy-path transition list from run_full_pipeline
# ---------------------------------------------------------------------------

def _full_pipeline_fire_all(net: PetriNet):
    def tokens(place: str) -> list:
        p = net.places.get(place)
        return list(p.marking) if p is not None else []

    def _try(tname: str, b: Binding):
        if net.is_enabled(tname, b):
            yield (tname, b)

    layout_toks = tokens("IA_input_layout")
    topo_toks   = tokens("IA_topology")
    vbuf_toks   = tokens("IA_vertex_buffer")
    for v, layout, topo in itertools.product(vbuf_toks, layout_toks, topo_toks):
        yield from _try("IA_fetch_vertex", {"v": v, "layout": layout, "topo": topo})

    vs_shader = tokens("VS_shader")
    vs_cbuf   = tokens("VS_cbuffer")
    for lane in range(4):
        for v, slot in itertools.product(tokens("VS_input"), tokens(f"VS_lane_{lane}_slot")):
            yield from _try(f"VS_dispatch_lane_{lane}", {"v": v, "slot": slot})
        for v, s, c in itertools.product(tokens(f"VS_lane_{lane}_input"), vs_shader, vs_cbuf):
            x, y, z, w = v.clip_pos
            v_out = Vertex(
                vertex_id=v.vertex_id,
                clip_pos=(x * 0.5, y, z, w),
                instance_id=v.instance_id,
                attributes=v.attributes,
            )
            yield from _try(f"VS_exec_lane_{lane}", {"v": v, "cbuf": c, "vs_shader": s, "v_out": v_out})
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
                yield from _try("RS_rasterize_triangle", {"prim": prim, "rs": rs, "cov": cov, "fill": fill})
        elif topo == "line_list":
            for laa in laa_toks:
                yield from _try("RS_rasterize_line", {"prim": prim, "laa": laa})
        else:
            yield from _try("RS_rasterize_point", {"prim": prim})

    ps_shader = tokens("PS_shader")
    ps_cbuf   = tokens("PS_cbuffer")
    for lane in range(4):
        for frag, slot in itertools.product(tokens("PS_input"), tokens(f"PS_lane_{lane}_slot")):
            yield from _try(f"PS_dispatch_lane_{lane}", {"frag": frag, "slot": slot})
        for frag, s, c in itertools.product(tokens(f"PS_lane_{lane}_input"), ps_shader, ps_cbuf):
            px = Pixel(
                pixel_x=frag.pixel_x, pixel_y=frag.pixel_y,
                color=(1.0, 0.0, 0.0, 1.0), depth=frag.depth, output_slot=0,
            )
            yield from _try(f"PS_exec_lane_{lane}", {"frag": frag, "cbuf": c, "ps_shader": s, "px": px})
        for px in tokens(f"PS_lane_{lane}_output"):
            yield from _try(f"PS_collect_lane_{lane}", {"px": px, "slot": LaneSlot(lane=lane)})

    bs_toks  = tokens("OM_blend_state")
    dss_toks = tokens("OM_depth_stencil_state")
    dsv_toks = tokens("OM_dsv")
    ra_toks  = tokens("OM_rt_active")
    for px in tokens("PS_output"):
        slot   = px.output_slot
        rt_toks = tokens(f"OM_rt_slot_{slot}")
        for rt, bs, ra in itertools.product(rt_toks, bs_toks, ra_toks):
            yield from _try(f"OM_write_rt_{slot}", {"px": px, "rt": rt, "bs": bs, "ra": ra})
        for dss, rt in itertools.product(dss_toks, dsv_toks):
            yield from _try("OM_depth_test", {"px": px, "dss": dss, "rt": rt})
    for ra in tokens("OM_rt_in_use"):
        yield from _try("OM_release_rt", {"ra": ra})


_HAPPY_PATH_TRANSITIONS: tuple[str, ...] = (
    "IA_fetch_vertex",
    *(f"VS_dispatch_lane_{l}" for l in range(4)),
    *(f"VS_exec_lane_{l}"     for l in range(4)),
    *(f"VS_collect_lane_{l}"  for l in range(4)),
    "HS_bypass_exec", "GS_bypass_exec", "SO_bypass_exec",
    "RS_clip_near", "RS_rasterize_triangle", "RS_rasterize_line", "RS_rasterize_point",
    *(f"PS_dispatch_lane_{l}" for l in range(4)),
    *(f"PS_exec_lane_{l}"     for l in range(4)),
    *(f"PS_collect_lane_{l}"  for l in range(4)),
    "OM_write_rt_0", "OM_depth_test", "OM_release_rt",
)


# ---------------------------------------------------------------------------
# Per-engine run helper
# ---------------------------------------------------------------------------

def _seed_net(n_vertices: int) -> PetriNet:
    net = build_full_pipeline()
    for i in range(n_vertices):
        net.places["IA_vertex_buffer"].add_token(
            Vertex(
                vertex_id=i,
                clip_pos=(float(i + 1), float(i), 0.5, 1.0),
                instance_id=0,
                attributes=(),
            )
        )
    return net


def _digest(m: FrozenMarking) -> bytes:
    """128-bit digest of a marking from a canonical (sorted repr) encoding.
    Tokens are frozen dataclasses with deterministic reprs, so equal markings
    from either engine get equal digests.  Comparing digests instead of keeping
    both N=4 state spaces alive keeps memory at one state space (two ~4 GB
    graphs at once push a 16 GB machine into swap)."""
    return hashlib.blake2b(repr(sorted(repr(e) for e in m)).encode(), digest_size=16).digest()


def _run_engine(n_vertices: int, engine: str, max_states: int) -> dict:
    net = _seed_net(n_vertices)
    inputs = _input_places(net)
    t0 = time.perf_counter()
    if engine == "snakes":
        snk_net, dropped = translate_to_snakes(net)
        ss = snakes_to_state_space(snk_net, max_states=max_states, transitions=_HAPPY_PATH_TRANSITIONS)
    else:
        ss = build_state_space(net, _full_pipeline_fire_all, max_states=max_states)
    bfs_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    ltl_results = verify_dx11_properties(ss)          # L3 = stutter notion
    ltl_s = time.perf_counter() - t1
    l3_strict = LTLChecker().check(ss, _build_p3(dead_states(ss, "strict"))).satisfied

    dg = {m: _digest(m) for m in ss.states}
    # (marking, transition, successor) -> number of firings the engine enumerated
    triples = Counter((dg[m], t, dg[d]) for m, outs in ss.edges.items() for t, d in outs)
    # For triples fired more than once, the multiplicity that identical-token
    # duplication predicts: product over t's input places of the largest token
    # multiplicity there (exact when each input place holds one distinct token).
    expected_mult: dict = {}
    for m, outs in ss.edges.items():
        for t, d in outs:
            key = (dg[m], t, dg[d])
            if triples[key] > 1 and key not in expected_mult:
                e = 1
                for p in inputs[t]:
                    counts = [cnt for pn, _tok, cnt in m if pn == p]
                    e *= max(counts) if counts else 1
                expected_mult[key] = e
    out = {
        "states":    len(ss.states),
        "state_set": set(dg.values()),
        "edges":     sum(triples.values()),
        "triples":   triples,
        "n_triples": len(triples),
        "pairs":     {(a, b) for (a, _t, b) in triples},
        "deadlocks": len(ss.deadlock_states()),
        "stutter_dead": len(ss.stutter_dead_states()),
        "truncated": ss.truncated,
        "bfs_s":     bfs_s,
        "ltl_s":     ltl_s,
        "ltl_pass":  {k: r.satisfied for k, r in ltl_results.items()},
        "l3_strict": l3_strict,
        "expected_mult": expected_mult,
    }
    out["n_pairs"] = len(out["pairs"])
    del ss, dg
    gc.collect()
    return out


def _input_places(net: PetriNet) -> dict[str, list[str]]:
    return {t: sorted({a.place.name for a in net.input_arcs(net.transitions[t])})
            for t in _HAPPY_PATH_TRANSITIONS if t in net.transitions}


def _explain_extra_firings(n: int, c: dict, s: dict) -> None:
    """Characterise firings SNAKES enumerates beyond the custom engine.

    SNAKES' Transition.modes() yields one substitution per token *occurrence*,
    so a place holding k copies of an identical token yields k identical
    modes -- k firings of the same (m, t, m') triple.  The custom engine
    iterates over distinct tokens.  A differing triple counts as explained when
    custom fires it once and SNAKES fires it exactly the identical-token
    multiplicity computed in _run_engine."""
    by_t: Counter = Counter()          # extra firings per transition
    mult_hist: Counter = Counter()     # (t, snakes_mult, custom_mult) -> #triples
    explained = unexplained = 0
    for key, ks in s["triples"].items():
        kc = c["triples"].get(key, 0)
        if ks == kc:
            continue
        t = key[1]
        by_t[t] += ks - kc
        mult_hist[(t, ks, kc)] += 1
        if kc == 1 and ks == s["expected_mult"].get(key):
            explained += 1
        else:
            unexplained += 1
    only_c = sum(1 for k in c["triples"] if k not in s["triples"])
    print(f"     extra SNAKES firings by transition: {dict(by_t)}  (total {sum(by_t.values())})")
    print(f"     triples with differing multiplicity (t, snakes, custom) -> count: {dict(mult_hist)}")
    print(f"     differing triples explained by identical-token multiplicity: "
          f"{explained}; unexplained: {unexplained}; triples only in custom: {only_c}")
    print(f"     custom triples fired more than once: "
          f"{sum(1 for v in c['triples'].values() if v > 1)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min"


def run_crossvalidation(targets: list[int], max_states: int) -> None:
    header = (f"{'N':>3}  {'Engine':<8}  {'States':>8}  {'Firings':>8}  {'(m,t,m)':>8}  "
              f"{'(m,m)':>8}  {'Dead':>4}  {'SDead':>5}  {'BFS':>10}  {'LTL':>10}  {'All pass':>8}  L3strict")
    sep    = "-" * len(header)

    print(f"\n{'='*60}")
    print("  DX11 Full-Pipeline Cross-Validation: custom vs. SNAKES")
    print(f"{'='*60}")
    print(header)

    mismatches: list[str] = []

    for n in targets:
        print(sep)
        results = {}
        for engine in ("custom", "snakes"):
            print(f"  Running N={n}, engine={engine} ...", end="", flush=True)
            r = _run_engine(n, engine, max_states)
            results[engine] = r
            all_pass = all(r["ltl_pass"].values())
            trunc    = " [TRUNCATED]" if r["truncated"] else ""
            print(f"\r{n:>3}  {engine:<8}  {r['states']:>8}  {r['edges']:>8}  "
                  f"{r['n_triples']:>8}  {r['n_pairs']:>8}  "
                  f"{r['deadlocks']:>4}  {r['stutter_dead']:>5}  {_fmt_time(r['bfs_s']):>10}  "
                  f"{_fmt_time(r['ltl_s']):>10}  {'YES' if all_pass else 'NO':>8}  "
                  f"{'PASS' if r['l3_strict'] else 'FAIL'}{trunc}", flush=True)

        # Compare
        c, s = results["custom"], results["snakes"]
        if c["states"] != s["states"]:
            mismatches.append(f"N={n}: state count mismatch  custom={c['states']}  snakes={s['states']}")
        if c["state_set"] != s["state_set"]:
            mismatches.append(f"N={n}: state SETS differ")
        if c["pairs"] != s["pairs"]:
            mismatches.append(f"N={n}: distinct (m,m') pair sets differ  "
                              f"custom-only={len(c['pairs'] - s['pairs'])}  "
                              f"snakes-only={len(s['pairs'] - c['pairs'])}")
        if set(c["triples"]) != set(s["triples"]):
            mismatches.append(f"N={n}: distinct (m,t,m') triple sets differ")
        if c["stutter_dead"] != s["stutter_dead"]:
            mismatches.append(f"N={n}: stutter-dead count mismatch")
        if c["deadlocks"] != s["deadlocks"]:
            mismatches.append(f"N={n}: deadlock count mismatch  custom={c['deadlocks']}  snakes={s['deadlocks']}")
        for prop in c["ltl_pass"]:
            if c["ltl_pass"][prop] != s["ltl_pass"].get(prop):
                mismatches.append(f"N={n}: LTL {prop} disagrees  custom={c['ltl_pass'][prop]}  snakes={s['ltl_pass'].get(prop)}")
        print(f"     state sets equal: {c['state_set'] == s['state_set']};  "
              f"distinct (m,m') pair sets equal: {c['pairs'] == s['pairs']};  "
              f"distinct (m,t,m') triple sets equal: {set(c['triples']) == set(s['triples'])}")
        if c["edges"] != s["edges"]:
            print(f"     raw firing counts differ: custom={c['edges']}, snakes={s['edges']} "
                  f"(+{s['edges'] - c['edges']})")
            _explain_extra_firings(n, c, s)
        print(f"     {'MATCH ✓' if not any(f'N={n}:' in m for m in mismatches) else 'MISMATCH ✗'}", flush=True)
        del results, c, s
        gc.collect()

    print(sep)
    print()
    if mismatches:
        print("MISMATCHES DETECTED:")
        for m in mismatches:
            print(f"  ✗  {m}")
        sys.exit(1)
    else:
        print("All cross-validation checks passed — state counts, deadlock counts, and LTL outcomes agree.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=None,
                        help="Run a single N instead of the default range")
    parser.add_argument("--include-n4", action="store_true",
                        help="Include N=4 in the SNAKES run (expected ~2.5 hours)")
    parser.add_argument("--max-states", type=int, default=300_000,
                        help="BFS state-count limit per engine (default: 300000)")
    args = parser.parse_args()

    if args.n is not None:
        targets = [args.n]
    elif args.include_n4:
        targets = [1, 2, 3, 4]
    else:
        targets = [1, 2, 3]

    if 4 in targets and not args.include_n4 and args.n != 4:
        targets = [t for t in targets if t != 4]

    run_crossvalidation(targets, args.max_states)


if __name__ == "__main__":
    main()
