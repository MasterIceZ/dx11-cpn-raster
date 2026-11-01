
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from pipeline.petri.net import Binding, PetriNet







FrozenMarking = frozenset[tuple[str, Any, int]]



FireAll = Callable[[PetriNet], Iterable[tuple[str, Binding]]]

def freeze(net: PetriNet) -> FrozenMarking:
    return frozenset(
        (name, token, count)
        for name, place in net.places.items()
        for token, count in place.marking.items()
        if count > 0
    )

def thaw(net: PetriNet, fm: FrozenMarking) -> None:
    for place in net.places.values():
        place.marking.clear()
    for place_name, token, count in fm:
        net.places[place_name].marking[token] = count

def place_tokens(fm: FrozenMarking, place_name: str) -> dict[Any, int]:
    return {tok: cnt for pn, tok, cnt in fm if pn == place_name and cnt > 0}

def place_count(fm: FrozenMarking, place_name: str) -> int:
    return sum(cnt for pn, _tok, cnt in fm if pn == place_name)





@dataclass
class StateSpace:

    initial: FrozenMarking
    states: set[FrozenMarking] = field(default_factory=set)
    edges: dict[FrozenMarking, list[tuple[str, FrozenMarking]]] = field(
        default_factory=dict
    )
    truncated: bool = False

    def successors(self, m: FrozenMarking) -> list[tuple[str, FrozenMarking]]:
        return self.edges.get(m, [])

    def is_deadlock(self, m: FrozenMarking) -> bool:
        """Strict (Petri-net) deadlock: no enabled firing at all."""
        return not self.edges.get(m)

    def deadlock_states(self) -> set[FrozenMarking]:
        """States with no successor.  Kept for callers that need the classic
        notion (e.g. counting terminal markings); L3 uses
        :meth:`stutter_dead_states`."""
        return {m for m in self.states if self.is_deadlock(m)}

    def is_stutter_dead(self, m: FrozenMarking) -> bool:
        """No successor other than itself: no enabled firing, or every enabled
        firing is a self-loop (successor == m).  A cycle through two or more
        markings that makes no progress is *not* caught here.  A net-zero
        transition such as
        OM_depth_test can stay enabled forever on a stranded token without
        changing the marking; under the stutter rule the path properties
        already apply (Section III-F), such a state makes no progress and is
        as dead as one with nothing enabled."""
        return all(dst == m for _, dst in self.edges.get(m, ()))

    def stutter_dead_states(self) -> set[FrozenMarking]:
        return {m for m in self.states if self.is_stutter_dead(m)}

    # -- edge statistics ---------------------------------------------------

    def num_firings(self) -> int:
        """Raw edge count: one per enumerated (marking, transition, binding)."""
        return sum(len(v) for v in self.edges.values())

    def distinct_labelled_edges(self) -> set[tuple[FrozenMarking, str, FrozenMarking]]:
        """Distinct (marking, transition, successor) triples."""
        return {(m, t, d) for m, outs in self.edges.items() for t, d in outs}

    def distinct_successor_pairs(self) -> set[tuple[FrozenMarking, FrozenMarking]]:
        """Distinct (marking, successor-marking) pairs: the reachability graph
        as a relation, which is all AF, SCC and reachability analyses see."""
        return {(m, d) for m, outs in self.edges.items() for _, d in outs}

    def edge_stats(self) -> dict[str, int]:
        pairs = self.distinct_successor_pairs()
        return {
            "firings": self.num_firings(),
            "distinct_triples": len(self.distinct_labelled_edges()),
            "distinct_pairs": len(pairs),
            "distinct_pairs_no_selfloop": sum(1 for m, d in pairs if m != d),
            "selfloop_firings": sum(
                1 for m, outs in self.edges.items() for _, d in outs if d == m
            ),
        }

    def reachable_from(self, m: FrozenMarking) -> set[FrozenMarking]:
        visited: set[FrozenMarking] = set()
        queue: deque[FrozenMarking] = deque([m])
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            for _, nxt in self.successors(cur):
                queue.append(nxt)
        return visited

    def sccs(self) -> list[set[FrozenMarking]]:

        visited: set[FrozenMarking] = set()
        finish_order: list[FrozenMarking] = []

        for start in self.states:
            if start in visited:
                continue
            stack: list[tuple[FrozenMarking, Any]] = [
                (start, iter(self.successors(start)))
            ]
            visited.add(start)
            while stack:
                node, children = stack[-1]
                try:
                    _, child = next(children)
                    if child not in visited:
                        visited.add(child)
                        stack.append((child, iter(self.successors(child))))
                except StopIteration:
                    finish_order.append(node)
                    stack.pop()


        reverse: dict[FrozenMarking, list[FrozenMarking]] = {
            m: [] for m in self.states
        }
        for src in self.states:
            for _, dst in self.successors(src):
                reverse[dst].append(src)


        visited2: set[FrozenMarking] = set()
        result: list[set[FrozenMarking]] = []
        for start in reversed(finish_order):
            if start in visited2:
                continue
            scc: set[FrozenMarking] = set()
            dfs: list[FrozenMarking] = [start]
            while dfs:
                node = dfs.pop()
                if node in visited2:
                    continue
                visited2.add(node)
                scc.add(node)
                dfs.extend(reverse[node])
            result.append(scc)

        return result





def build_state_space(
    net: PetriNet,
    fire_all: FireAll,
    max_states: int = 10_000,
) -> StateSpace:
    initial = freeze(net)
    ss = StateSpace(initial=initial)
    ss.states.add(initial)

    queue: deque[FrozenMarking] = deque([initial])

    while queue:
        src_fm = queue.popleft()




        thaw(net, src_fm)
        firings: list[tuple[str, Binding]] = list(fire_all(net))

        src_edges: list[tuple[str, FrozenMarking]] = []

        for trans_name, binding in firings:
            thaw(net, src_fm)
            try:
                net.fire(trans_name, binding)
            except (RuntimeError, ValueError, OverflowError):
                continue

            dst_fm = freeze(net)
            src_edges.append((trans_name, dst_fm))

            if dst_fm not in ss.states:
                if len(ss.states) >= max_states:
                    ss.truncated = True
                    ss.states.add(dst_fm)
                    ss.edges[src_fm] = src_edges
                    thaw(net, initial)
                    return ss
                ss.states.add(dst_fm)
                queue.append(dst_fm)

        if src_edges:
            ss.edges[src_fm] = src_edges

    thaw(net, initial)
    return ss
