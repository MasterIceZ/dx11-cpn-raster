
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal





Binding = dict[str, Any]
ArcExpr = Callable[[Binding], "Multiset[Any]"]
Guard = Callable[[Binding], bool]





class Multiset(Counter[Any]):

    def __ge__(self, other: "Multiset[Any]") -> bool:
        return all(self[k] >= v for k, v in other.items())

    def __le__(self, other: "Multiset[Any]") -> bool:
        return other >= self

    def add(self, ms: "Multiset[Any]") -> None:
        for token, count in ms.items():
            self[token] += count

    def remove(self, ms: "Multiset[Any]") -> None:
        if not (self >= ms):
            missing = {k: ms[k] - self[k] for k in ms if ms[k] > self[k]}
            raise ValueError(f"Insufficient tokens: {missing}")
        for token, count in ms.items():
            self[token] -= count
            if self[token] == 0:
                del self[token]

    @classmethod
    def of(cls, *tokens: Any) -> "Multiset[Any]":
        return cls(tokens)





@dataclass
class Place:

    name: str
    color_type: type
    marking: Multiset[Any] = field(default_factory=Multiset)
    bound: int | None = None

    def add_token(self, token: Any) -> None:
        if not isinstance(token, self.color_type):
            raise TypeError(
                f"Place '{self.name}' expects {self.color_type.__name__}, "
                f"got {type(token).__name__}"
            )
        if self.bound is not None and self.token_count() >= self.bound:
            raise OverflowError(
                f"Place '{self.name}' is {self.bound}-bounded; cannot add token"
            )
        self.marking[token] += 1

    def remove_token(self, token: Any) -> None:
        if self.marking[token] <= 0:
            raise ValueError(f"Token {token!r} not present in place '{self.name}'")
        self.marking[token] -= 1
        if self.marking[token] == 0:
            del self.marking[token]

    def token_count(self) -> int:
        return sum(self.marking.values())

    def __repr__(self) -> str:
        return f"Place({self.name!r}, marking={dict(self.marking)})"





@dataclass
class Transition:

    name: str
    guard: Guard = field(default_factory=lambda: lambda _: True)

    def __repr__(self) -> str:
        return f"Transition({self.name!r})"





@dataclass
class Arc:

    place: Place
    transition: Transition
    direction: Literal["in", "out"]
    expression: ArcExpr

    def evaluate(self, binding: Binding) -> Multiset[Any]:
        return self.expression(binding)

    def __repr__(self) -> str:
        arrow = "→" if self.direction == "in" else "←"
        return f"Arc({self.place.name} {arrow} {self.transition.name})"





@dataclass
class PetriNet:

    name: str
    places: dict[str, Place] = field(default_factory=dict)
    transitions: dict[str, Transition] = field(default_factory=dict)
    arcs: list[Arc] = field(default_factory=list)





    def add_place(
        self,
        name: str,
        color_type: type,
        initial: Iterable[Any] | None = None,
        bound: int | None = None,
    ) -> Place:
        if name in self.places:
            raise ValueError(f"Place '{name}' already exists")
        p = Place(name=name, color_type=color_type, bound=bound)
        if initial is not None:
            for token in initial:
                p.add_token(token)
        self.places[name] = p
        return p

    def add_transition(
        self,
        name: str,
        guard: Guard | None = None,
    ) -> Transition:
        if name in self.transitions:
            raise ValueError(f"Transition '{name}' already exists")
        t = Transition(name=name, guard=guard or (lambda _: True))
        self.transitions[name] = t
        return t

    def add_arc(
        self,
        place: str | Place,
        transition: str | Transition,
        direction: Literal["in", "out"],
        expression: ArcExpr,
    ) -> Arc:
        p = self.places[place] if isinstance(place, str) else place
        t = self.transitions[transition] if isinstance(transition, str) else transition
        arc = Arc(place=p, transition=t, direction=direction, expression=expression)
        self.arcs.append(arc)
        return arc

    def add_input_arc(
        self,
        place: str | Place,
        transition: str | Transition,
        expression: ArcExpr,
    ) -> Arc:
        return self.add_arc(place, transition, "in", expression)

    def add_output_arc(
        self,
        place: str | Place,
        transition: str | Transition,
        expression: ArcExpr,
    ) -> Arc:
        return self.add_arc(place, transition, "out", expression)





    def input_arcs(self, transition: Transition) -> list[Arc]:
        return [a for a in self.arcs if a.transition is transition and a.direction == "in"]

    def output_arcs(self, transition: Transition) -> list[Arc]:
        return [a for a in self.arcs if a.transition is transition and a.direction == "out"]





    def is_enabled(self, transition: str | Transition, binding: Binding) -> bool:
        t = self._resolve_transition(transition)
        if not t.guard(binding):
            return False
        for arc in self.input_arcs(t):
            if not (arc.place.marking >= arc.evaluate(binding)):
                return False
        return True

    def fire(self, transition: str | Transition, binding: Binding) -> None:
        t = self._resolve_transition(transition)
        if not self.is_enabled(t, binding):
            raise RuntimeError(
                f"Transition '{t.name}' is not enabled under binding {binding}"
            )
        for arc in self.input_arcs(t):
            arc.place.marking.remove(arc.evaluate(binding))
        for arc in self.output_arcs(t):
            produced = arc.evaluate(binding)
            for token, count in produced.items():
                for _ in range(count):
                    arc.place.add_token(token)

    def enabled_transitions(self, binding: Binding) -> list[Transition]:
        return [t for t in self.transitions.values() if self.is_enabled(t, binding)]





    def merge(self, other: "PetriNet", *, name: str | None = None) -> "PetriNet":
        merged = PetriNet(name=name or f"{self.name}+{other.name}")

        for p in self.places.values():
            merged.places[p.name] = copy.deepcopy(p)
        for t in self.transitions.values():
            merged.transitions[t.name] = copy.deepcopy(t)

        for p in other.places.values():
            if p.name not in merged.places:
                merged.places[p.name] = copy.deepcopy(p)
        for t in other.transitions.values():
            if t.name not in merged.transitions:
                merged.transitions[t.name] = copy.deepcopy(t)

        for arc in self.arcs + other.arcs:
            merged.arcs.append(Arc(
                place=merged.places[arc.place.name],
                transition=merged.transitions[arc.transition.name],
                direction=arc.direction,
                expression=arc.expression,
            ))

        return merged





    def marking_snapshot(self) -> dict[str, dict[Any, int]]:
        return {name: dict(p.marking) for name, p in self.places.items()}

    def _resolve_transition(self, t: str | Transition) -> Transition:
        if isinstance(t, str):
            if t not in self.transitions:
                raise KeyError(f"No transition named '{t}'")
            return self.transitions[t]
        return t

    def __repr__(self) -> str:
        return (
            f"PetriNet({self.name!r}, "
            f"places={list(self.places)}, "
            f"transitions={list(self.transitions)})"
        )
