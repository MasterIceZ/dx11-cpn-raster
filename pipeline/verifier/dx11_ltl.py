
from __future__ import annotations

from typing import Any

from pipeline.petri.ltl import (
    And,
    Atom,
    CheckResult,
    F,
    G,
    Implies,
    LTLChecker,
    Not,
    Or,
    Formula,
)
from pipeline.petri.state_space import (
    FrozenMarking,
    StateSpace,
    place_count,
    place_tokens,
)







def _ia_has_input(m: FrozenMarking) -> bool:
    return place_count(m, "IA_output") > 0 or place_count(m, "VS_input") > 0

def _om_wrote_pixel(m: FrozenMarking) -> bool:
    return (
        any(place_count(m, f"OM_written_pixel_{i}") > 0 for i in range(8))
        or place_count(m, "OM_depth_buffer") > 0
        or place_count(m, "OM_stencil_buffer") > 0
    )

def _pixel_discarded(m: FrozenMarking) -> bool:
    return (
        place_count(m, "RS_culled_primitive") > 0
        or place_count(m, "PS_discard_signal") > 0
    )



_WRITTEN_SINKS: tuple[str, ...] = tuple(f"OM_written_pixel_{i}" for i in range(8))

# Stream-output capture is a sanctioned DX11 termination: a primitive written to
# an SO buffer has not reached the OM but has not been lost either.
_CAPTURED_SINKS: tuple[str, ...] = tuple(f"SO_stream_{i}" for i in range(4))

# Sanctioned discards.  PS_discard_signal is a *resource* place, not a sink, and
# was previously miscounted here; the fragment/pixel sinks below are the places
# the discard transitions actually deposit into.
_DISCARD_SINKS: tuple[str, ...] = (
    "RS_culled_primitive",
    "PS_discarded_fragment",
    "OM_discarded_pixel",
)

def _written_count(m: FrozenMarking) -> int:
    return sum(place_count(m, p) for p in _WRITTEN_SINKS)

def _captured_count(m: FrozenMarking) -> int:
    return sum(place_count(m, p) for p in _CAPTURED_SINKS)

def _discarded_count(m: FrozenMarking) -> int:
    return sum(place_count(m, p) for p in _DISCARD_SINKS)


def _concurrent_rt_write(m: FrozenMarking) -> bool:
    # OM_rt_in_use holds the render-target context token from OM_write_rt_i
    # until OM_release_rt.  Two tokens here means two render-target writes are
    # in flight at once -- the hazard the single OM_rt_active semaphore must
    # prevent when several fragments for the same pixel are pending.
    return place_count(m, "OM_rt_in_use") >= 2




_PIPELINE_BUFFERS: tuple[str, ...] = (
    "VS_input",
    "VS_output",
    "HS_input",
    "HS_output",
    "Tess_output",
    "DS_patch_input",
    "DS_output",
    "GS_input",
    "GS_output",
    "SO_input",
    # SO_stream_0..3 are capture sinks (nothing consumes them; L1 counts them
    # as captured), so a token there is finished work, not buffered work.
    "RS_input",
    "RS_clipped_primitive",
    "PS_input",
    "PS_output",
)

def _any_buffer_nonempty(m: FrozenMarking) -> bool:
    return any(place_count(m, p) > 0 for p in _PIPELINE_BUFFERS)




RS_QUEUE_BOUND: int = 1024

def _rs_queue_within_bound(m: FrozenMarking) -> bool:
    # PS_input is the rasterizer's fragment output queue: RS deposits
    # rasterized Fragment tokens here and the four PS lanes drain them via
    # PS_dispatch_lane_*. This is the model's actual "RS fragment queue".
    return place_count(m, "PS_input") <= RS_QUEUE_BOUND




def _om_rt_ready(m: FrozenMarking) -> bool:
    return place_count(m, "OM_rt_active") > 0



def _ps_waiting(m: FrozenMarking) -> bool:
    return place_count(m, "PS_mem_wait") > 0

def _mem_free(m: FrozenMarking) -> bool:
    return place_count(m, "MEM_free") > 0

def _ps_has_mem(m: FrozenMarking) -> bool:
    return place_count(m, "PS_mem_acquired") > 0







def _build_p1(n_vertices: int) -> Formula:
    # Token conservation: every seeded vertex ends in a written-pixel sink or a
    # sanctioned discard sink.  Stronger than "some pixel is eventually written",
    # which a single silently lost vertex would still satisfy.
    def _all_accounted(m: FrozenMarking) -> bool:
        return (_written_count(m) + _captured_count(m)
                + _discarded_count(m)) == n_vertices

    return G(
        Implies(
            Atom(_ia_has_input, "IA_has_input"),
            F(Atom(_all_accounted, f"written+captured+discarded={n_vertices}")),
        )
    )



P2_OM_MUTEX: Formula = G(Not(Atom(_concurrent_rt_write, "concurrent_rt_write")))



P4_RS_BOUNDED: Formula = G(Atom(_rs_queue_within_bound, f"rs_queue_≤{RS_QUEUE_BOUND}"))



P5_PS_NO_STARVATION: Formula = G(
    Implies(
        And(Atom(_ps_waiting, "PS_waiting"), Atom(_mem_free, "mem_free")),
        F(Atom(_ps_has_mem, "PS_has_mem")),
    )
)


def _lane_in(place: str, lane: int):
    """True iff `place` holds a lane-tagged token for this particular lane."""
    def pred(m: FrozenMarking) -> bool:
        return any(getattr(tok, "lane", None) == lane
                   for tok in place_tokens(m, place))
    return pred


def p5_per_lane(lane: int) -> Formula:
    """Indexed form of L5: *this* lane, once waiting with the port free, must
    itself acquire the port.  The aggregate form P5_PS_NO_STARVATION quantifies
    only over the marking ("some lane holds the port"), so it cannot detect one
    lane starving while its neighbours cycle; instantiating this formula for
    every lane recovers the intended per-thread reading over the finite,
    statically known lane set."""
    return G(
        Implies(
            And(Atom(_lane_in("PS_mem_wait", lane), f"lane{lane}_waiting"),
                Atom(_mem_free, "mem_free")),
            F(Atom(_lane_in("PS_mem_acquired", lane), f"lane{lane}_has_mem")),
        )
    )





P6_RT_AVAILABLE: Formula = G(F(Atom(_om_rt_ready, "om_rt_ready")))

def _build_p3(deadlock_states: frozenset[FrozenMarking]) -> Formula:
    def _not_deadlock(m: FrozenMarking) -> bool:
        return m not in deadlock_states

    return G(
        Implies(
            Atom(_any_buffer_nonempty, "any_buffer_nonempty"),
            Atom(_not_deadlock, "¬deadlock"),
        )
    )





DEADLOCK_NOTIONS: tuple[str, ...] = ("stutter", "strict")


def dead_states(ss: StateSpace, notion: str = "stutter") -> frozenset[FrozenMarking]:
    """States L3 treats as dead.

    ``"stutter"`` (default): no successor, or only self-loop successors -- the
    same stutter rule the path properties use, so a hang kept alive by a
    net-zero transition (OM_depth_test on a stranded pixel) counts as dead.
    ``"strict"``: no successor at all (the notion used up to the round-3
    camera-ready, kept for comparison)."""
    if notion == "stutter":
        return frozenset(ss.stutter_dead_states())
    if notion == "strict":
        return frozenset(ss.deadlock_states())
    raise ValueError(f"unknown deadlock notion {notion!r}; use one of {DEADLOCK_NOTIONS}")


def verify_dx11_properties(
    ss: StateSpace,
    *,
    rs_queue_bound: int = RS_QUEUE_BOUND,
    deadlock: str = "stutter",
) -> dict[str, CheckResult]:
    checker = LTLChecker()
    deadlocks = dead_states(ss, deadlock)
    n_vertices = place_count(ss.initial, "IA_vertex_buffer")

    p4_formula: Formula = G(
        Atom(
            lambda m: place_count(m, "PS_input") <= rs_queue_bound,
            f"rs_queue_≤{rs_queue_bound}",
        )
    )

    p3_formula = _build_p3(deadlocks)
    if ss.truncated:
        # Unexpanded frontier states have no recorded successors (and the state
        # being expanded at the cutoff may have only part of its edges), so
        # either notion would report them as dead.
        p3 = CheckResult(
            p3_formula, False,
            note="inconclusive: state space truncated, deadlock undecidable at frontier",
        )
    else:
        p3 = checker.check(ss, p3_formula)

    return {
        "P1": checker.check(ss, _build_p1(n_vertices)),
        "P2": checker.check(ss, P2_OM_MUTEX),
        "P3": p3,
        "P4": checker.check(ss, p4_formula),
        "P5": checker.check(ss, P5_PS_NO_STARVATION),
        "P6": checker.check(ss, P6_RT_AVAILABLE),
    }
