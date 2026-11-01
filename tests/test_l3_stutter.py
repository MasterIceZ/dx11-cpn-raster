"""L3 deadlock notion (round-3 review W2) and edge statistics (W4)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from pipeline.petri.colored import Pixel
from pipeline.petri.net import Binding, Multiset, PetriNet
from pipeline.petri.state_space import StateSpace, build_state_space
from pipeline.verifier.dx11_ltl import dead_states, verify_dx11_properties

PX = Pixel(pixel_x=0, pixel_y=0, color=(1.0, 0.0, 0.0, 1.0), depth=0.5, output_slot=0)


def _tiny_net(*, with_drain: bool) -> PetriNet:
    """PS_output (one of L3's listed buffers) holding a pixel, a net-zero
    `peek` that reads it (a self-loop, like OM_depth_test), and optionally a
    `drain` that moves it to a sink (a real progress step)."""
    net = PetriNet(name="tiny")
    net.add_place("PS_output", Pixel, initial=[PX])
    net.add_place("sink", Pixel)
    net.add_transition("peek")
    net.add_input_arc("PS_output", "peek", lambda b: Multiset.of(b["px"]))
    net.add_output_arc("PS_output", "peek", lambda b: Multiset.of(b["px"]))
    if with_drain:
        net.add_transition("drain")
        net.add_input_arc("PS_output", "drain", lambda b: Multiset.of(b["px"]))
        net.add_output_arc("sink", "drain", lambda b: Multiset.of(b["px"]))
    return net


def _fire_all(net: PetriNet) -> Iterable[tuple[str, Binding]]:
    for px in list(net.places["PS_output"].marking):
        for t in ("peek", "drain"):
            if t in net.transitions and net.is_enabled(t, {"px": px}):
                yield (t, {"px": px})


def test_self_loop_only_state_is_dead() -> None:
    ss = build_state_space(_tiny_net(with_drain=False), _fire_all)
    assert len(ss.states) == 1
    assert ss.successors(ss.initial) == [("peek", ss.initial)]
    # Old notion: something is enabled, so not dead.  New notion: only stutter.
    assert ss.deadlock_states() == set()
    assert ss.stutter_dead_states() == {ss.initial}
    assert dead_states(ss, "strict") == frozenset()
    assert dead_states(ss, "stutter") == frozenset({ss.initial})


def test_l3_fails_on_hang_kept_alive_by_self_loop() -> None:
    ss = build_state_space(_tiny_net(with_drain=False), _fire_all)
    assert verify_dx11_properties(ss)["P3"].satisfied is False
    assert verify_dx11_properties(ss, deadlock="strict")["P3"].satisfied is True


def test_real_progress_step_is_not_dead() -> None:
    ss = build_state_space(_tiny_net(with_drain=True), _fire_all)
    assert len(ss.states) == 2
    # The initial state has a self-loop *and* a progress edge: not dead.
    assert not ss.is_stutter_dead(ss.initial)
    (terminal,) = ss.stutter_dead_states()
    assert terminal != ss.initial and ss.successors(terminal) == []
    # The only dead state has an empty PS_output, so L3 holds.
    assert verify_dx11_properties(ss)["P3"].satisfied is True


def test_unknown_notion_rejected() -> None:
    ss = build_state_space(_tiny_net(with_drain=False), _fire_all)
    with pytest.raises(ValueError):
        dead_states(ss, "bogus")


def test_edge_stats_count_distinct_pairs() -> None:
    a, b = frozenset({("p", 1, 1)}), frozenset({("q", 1, 1)})
    ss = StateSpace(initial=a, states={a, b},
                    edges={a: [("t", b), ("t", b), ("u", b), ("s", a)]})
    st = ss.edge_stats()
    assert st["firings"] == 4
    assert st["distinct_triples"] == 3          # (a,t,b) (a,u,b) (a,s,a)
    assert st["distinct_pairs"] == 2            # (a,b) (a,a)
    assert st["distinct_pairs_no_selfloop"] == 1
    assert st["selfloop_firings"] == 1
    assert ss.distinct_successor_pairs() == {(a, b), (a, a)}


# --- composed-net regressions --------------------------------------------

def test_composed_net_l3_holds_under_both_notions_n1() -> None:
    from run_crossvalidation import _seed_net
    from run_full_pipeline import _full_pipeline_fire_all
    ss = build_state_space(_seed_net(1), _full_pipeline_fire_all)
    assert (len(ss.states), ss.num_firings()) == (26, 32)
    assert ss.stutter_dead_states() == ss.deadlock_states()
    assert verify_dx11_properties(ss)["P3"].satisfied
    assert verify_dx11_properties(ss, deadlock="strict")["P3"].satisfied


def test_no_om_release_rt_fault_now_violates_l3() -> None:
    import run_mutation_tests as M
    out = M._m_l6(2)
    assert out["L6"] is False and out["L1"] is False
    assert out["L3"] is False          # stutter notion sees the stranded pixel
    assert out["L3strict"] is True     # the old notion did not


def test_l4_fanout_fault_is_caught_at_net_level() -> None:
    import run_mutation_tests as M
    assert M._m_l4_control(2)["L4"] is True
    assert M._m_l4_fanout(2)["L4"] is False


def test_snakes_and_custom_agree_on_distinct_pairs_n1() -> None:
    from pipeline.petri.snakes_bridge import snakes_to_state_space
    from pipeline.petri.to_snakes import translate
    from run_crossvalidation import _HAPPY_PATH_TRANSITIONS, _seed_net
    from run_full_pipeline import _full_pipeline_fire_all
    c = build_state_space(_seed_net(1), _full_pipeline_fire_all)
    snk, _ = translate(_seed_net(1))
    s = snakes_to_state_space(snk, max_states=10_000, transitions=_HAPPY_PATH_TRANSITIONS)
    assert c.states == s.states
    assert c.distinct_successor_pairs() == s.distinct_successor_pairs()
    assert c.distinct_labelled_edges() == s.distinct_labelled_edges()


def test_two_state_cycle_without_progress_is_not_dead() -> None:
    # Pins the documented limit of the rule: only single-state stutter counts.
    a, b = frozenset({("PS_output", 1, 1)}), frozenset({("PS_input", 1, 1)})
    ss = StateSpace(initial=a, states={a, b},
                    edges={a: [("t", b)], b: [("u", a)]})
    assert ss.stutter_dead_states() == set()


def test_l3_inconclusive_on_truncated_state_space() -> None:
    ss = build_state_space(_tiny_net(with_drain=True), _fire_all, max_states=1)
    assert ss.truncated
    res = verify_dx11_properties(ss)["P3"]
    assert res.satisfied is False and res.note.startswith("inconclusive")


def test_l1_does_not_detect_duplication() -> None:
    # Known limit reported in the paper: G(input -> F(accounted = N)) is met
    # when the count passes through N, so a fault that writes 2N pixels holds.
    import run_mutation_tests as M
    out = M._m_l4_fanout(2)
    assert out["L1"] is True
