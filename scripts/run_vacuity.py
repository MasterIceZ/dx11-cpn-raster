
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import run_full_pipeline` resolves

import argparse

from pipeline.petri.colored import Vertex
from pipeline.petri.state_space import build_state_space
from pipeline.verifier.dx11_ltl import (
    _ia_has_input,
    _any_buffer_nonempty,
    _ps_waiting,
    _mem_free,
    _om_rt_ready,
    _written_count,
    _discarded_count,
)
from models.dx11_full_pipeline import build_full_pipeline

from run_full_pipeline import DEFAULT_MAX_STATES, _full_pipeline_fire_all


def _bottom_sccs(ss):
    """Bottom SCCs (no outgoing edge to another SCC) — the targets a GF
    liveness witness must intersect.  Mirrors the logic in LTLChecker._check_finally."""
    sccs = ss.sccs()
    state_to_scc = {s: i for i, scc in enumerate(sccs) for s in scc}
    has_exit = [False] * len(sccs)
    for src in ss.states:
        si = state_to_scc[src]
        for _, dst in ss.successors(src):
            if state_to_scc[dst] != si:
                has_exit[si] = True
    return [scc for i, scc in enumerate(sccs) if not has_exit[i]]


def run(n: int, lanes: int = 4, max_states: int = DEFAULT_MAX_STATES) -> None:
    net = build_full_pipeline({"lanes": lanes})
    for i in range(n):
        net.places["IA_vertex_buffer"].add_token(
            Vertex(vertex_id=i, clip_pos=(float(i + 1), float(i), 0.5, 1.0))
        )
    ss = build_state_space(net, _full_pipeline_fire_all, max_states=max_states)
    deadlocks = ss.deadlock_states()          # strict: nothing enabled
    stutter_dead = ss.stutter_dead_states()   # L3's notion: only self-loops
    total = len(ss.states)

    print(f"\n{'='*72}")
    print(f"  Vacuity analysis on the composed pipeline  |  N = {n}  |  lanes = {lanes}  |  "
          f"{total} states  |  truncated = {ss.truncated}")
    print(f"{'='*72}")

    def verdict(v: str) -> str:
        # Frontier states have no recorded successors, so they look dead and form
        # singleton bottom SCCs: no verdict below is sound on a truncated space.
        return "INCONCLUSIVE (truncated)" if ss.truncated else v

    # --- L1: antecedent reachable & consequent genuinely reached? ----------
    l1_trigger = sum(1 for m in ss.states if _ia_has_input(m))
    l1_conseq = sum(1 for m in ss.states if _written_count(m) + _discarded_count(m) == n)
    print(f"  L1  ia_has_input holds in            {l1_trigger:>7}/{total} states "
          f"(consequent reached in {l1_conseq}) -> "
          f"{verdict('SUBSTANTIVE' if l1_trigger and l1_conseq else 'VACUOUS')}")

    # --- L3: buffer-nonempty at a NON-deadlock (live) state? ---------------
    l3_live = sum(1 for m in ss.states
                  if _any_buffer_nonempty(m) and m not in stutter_dead)
    l3_live_strict = sum(1 for m in ss.states
                         if _any_buffer_nonempty(m) and m not in deadlocks)
    print(f"  L3  any_buffer_nonempty at non-dead   {l3_live:>5}/{total} states "
          f"(stutter-dead states: {len(stutter_dead)}; strict-notion count {l3_live_strict}, "
          f"strict dead {len(deadlocks)}) -> "
          f"{verdict('SUBSTANTIVE (non-vacuous off the terminal state)' if l3_live else 'VACUOUS')}")

    # --- L4: trivially handled by stage-local table; report occupancy ------
    from pipeline.petri.state_space import place_count
    max_q = max(place_count(m, "PS_input") for m in ss.states)
    print(f"  L4  max |PS_input| (rasterizer queue) {max_q:>7}   (declared bound 1024)   -> "
          f"{verdict(f'STRUCTURAL SANITY (margin {1024 // max(max_q,1)}x)')}")

    # --- L5: trigger places exist at all in the composed net? --------------
    l5_trigger = sum(1 for m in ss.states if _ps_waiting(m) and _mem_free(m))
    have_places = ("PS_mem_wait" in net.places and "MEM_free" in net.places)
    print(f"  L5  (PS_waiting ∧ mem_free) holds in  {l5_trigger:>7}/{total} states "
          f"(trigger places present: {have_places}) -> "
          f"{verdict('SUBSTANTIVE' if l5_trigger else 'VACUOUS in composed net (see stage-local PS table)')}")

    # --- L6: does the witness sit only in the terminal SCC? ----------------
    bsccs = _bottom_sccs(ss)
    term_only = all(len(scc) == 1 and next(iter(scc)) in deadlocks for scc in bsccs)
    witness_in_bottom = any(_om_rt_ready(m) for scc in bsccs for m in scc)
    print(f"  L6  bottom SCCs: {len(bsccs)} (all terminal singletons: {term_only}); "
          f"om_rt_ready in a bottom SCC: {witness_in_bottom} -> "
          f"{verdict('WITNESSED ONLY AT TERMINAL STATE' if term_only else 'SUBSTANTIVE')}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-property vacuity analysis on the composed DX11 pipeline."
    )
    ap.add_argument("--n", type=int, default=3, help="input vertices (default: 3)")
    ap.add_argument("--lanes", type=int, default=4, help="VS/PS lanes (default: 4)")
    ap.add_argument("--max-states", type=int, default=DEFAULT_MAX_STATES,
                    help=f"BFS state-count limit (default: {DEFAULT_MAX_STATES})")
    args = ap.parse_args()
    run(args.n, lanes=args.lanes, max_states=args.max_states)


if __name__ == "__main__":
    main()
