
from __future__ import annotations

from collections import Counter, deque
from typing import Any, Iterable

from snakes.nets import (
    Marking,
    MultiSet as SNKMultiSet,
    PetriNet as SNKNet,
)

from pipeline.petri.state_space import FrozenMarking, StateSpace


# ---------------------------------------------------------------------------
# Marking conversion helpers
# ---------------------------------------------------------------------------

def freeze_snakes(net: SNKNet) -> FrozenMarking:
    """Convert a SNAKES net's current marking → our immutable FrozenMarking.

    SNAKES `Marking` only carries non-empty places, so we walk all places in
    the net to ensure empty places are captured (as zero entries, i.e., absent
    from the frozenset) consistently.
    """
    m = net.get_marking()
    entries: list[tuple[str, Any, int]] = []
    for place in net.place():
        ms = m.get(place.name)
        if ms is None:
            continue
        for token, count in Counter(ms).items():
            if count > 0:
                entries.append((place.name, token, count))
    return frozenset(entries)


def thaw_snakes(net: SNKNet, fm: FrozenMarking) -> None:
    """Restore a SNAKES net's marking from a FrozenMarking snapshot."""
    place_tokens: dict[str, list[Any]] = {p.name: [] for p in net.place()}
    for place_name, token, count in fm:
        if place_name in place_tokens:
            place_tokens[place_name].extend([token] * count)
    net.set_marking(
        Marking({name: SNKMultiSet(toks) for name, toks in place_tokens.items()})
    )


# ---------------------------------------------------------------------------
# State-space exploration using SNAKES mode enumeration
# ---------------------------------------------------------------------------

def snakes_to_state_space(
    net: SNKNet,
    max_states: int = 5_000,
    transitions: Iterable[str] | None = None,
) -> StateSpace:
    """Explore the reachable state space of a SNAKES PetriNet.

    Uses SNAKES' ``Transition.modes()`` for sound, complete binding enumeration
    instead of the manual cartesian-product approach in our custom engine.
    The returned :class:`StateSpace` is compatible with our LTL checker.

    Parameters
    ----------
    net:
        A SNAKES ``PetriNet`` whose initial marking is already set.
    max_states:
        BFS exploration cap; sets ``StateSpace.truncated`` if hit.
    transitions:
        If given, restrict exploration to this subset of transition names
        (e.g. the same progress-making subset enumerated by the custom
        engine's fire-all driver) rather than every transition in *net*.
        This keeps the two engines' reachability graphs comparable and
        avoids exploring read-only/loop transitions that the custom driver
        deliberately omits.
    """
    initial = freeze_snakes(net)
    ss = StateSpace(initial=initial)
    ss.states.add(initial)
    queue: deque[FrozenMarking] = deque([initial])

    if transitions is not None:
        all_names = {t.name for t in net.transition()}
        trans_names = [name for name in transitions if name in all_names]
    else:
        trans_names = [t.name for t in net.transition()]

    while queue:
        src_fm = queue.popleft()
        thaw_snakes(net, src_fm)

        src_edges: list[tuple[str, FrozenMarking]] = []

        for trans_name in trans_names:
            t = net.transition(trans_name)
            thaw_snakes(net, src_fm)
            try:
                modes = list(t.modes())
            except Exception:
                # Guard expression may raise if variables aren't bound properly
                continue

            for mode in modes:
                thaw_snakes(net, src_fm)
                try:
                    t.fire(mode)
                except Exception:
                    continue

                dst_fm = freeze_snakes(net)
                src_edges.append((trans_name, dst_fm))

                if dst_fm not in ss.states:
                    if len(ss.states) >= max_states:
                        ss.truncated = True
                        ss.states.add(dst_fm)
                        if src_edges:
                            ss.edges[src_fm] = src_edges
                        thaw_snakes(net, initial)
                        return ss
                    ss.states.add(dst_fm)
                    queue.append(dst_fm)

        if src_edges:
            ss.edges[src_fm] = src_edges

    thaw_snakes(net, initial)
    return ss
