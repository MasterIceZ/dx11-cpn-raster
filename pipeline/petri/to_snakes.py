"""Automatic translator: custom :class:`PetriNet` → SNAKES :class:`snakes.nets.PetriNet`.

Every place and arc in the custom net is reproduced in SNAKES.  Input arcs
become SNAKES ``Variable`` arcs whose name is discovered by probing the
original arc lambda.  Output arcs and guards wrap the original Python
lambdas behind SNAKES ``Expression`` objects that re-pack the SNAKES
substitution into the dict-shaped binding the custom engine expects.

Transitions whose outputs or guards reference variables that are not bound
by any input arc are dropped – SNAKES cannot synthesise an unbound output
variable, and such transitions likewise never fire in the custom engine's
fire-all exploration for the full DX11 pipeline.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from snakes.nets import (
    Expression,
    MultiArc,
    MultiSet as SNKMultiSet,
    PetriNet as SNKNet,
    Place as SNKPlace,
    Transition as SNKTransition,
    Variable,
)

from pipeline.petri.net import Binding, PetriNet


# ---------------------------------------------------------------------------
# Variable discovery
# ---------------------------------------------------------------------------

class _Sentinel:
    """Absorbing value that satisfies nearly every operation lambdas perform."""

    __slots__ = ()

    def __getattr__(self, _name):            return self
    def __getitem__(self, _key):              return self
    def __call__(self, *_a, **_kw):           return self
    def __bool__(self):                        return True
    def __iter__(self):                        return iter([self, self, self, self])
    def __len__(self):                         return 4
    def __contains__(self, _item):             return True
    def __int__(self):                         return 0
    def __float__(self):                       return 0.0
    def __index__(self):                       return 0
    def __hash__(self):                        return 0
    def __eq__(self, _other):                  return False
    def __ne__(self, _other):                  return True
    def __lt__(self, _other):                  return False
    def __le__(self, _other):                  return True
    def __gt__(self, _other):                  return False
    def __ge__(self, _other):                  return True
    def __add__(self, _o):                     return self
    def __radd__(self, _o):                    return self
    def __sub__(self, _o):                     return self
    def __rsub__(self, _o):                    return self
    def __mul__(self, _o):                     return self
    def __rmul__(self, _o):                    return self
    def __truediv__(self, _o):                 return self
    def __rtruediv__(self, _o):                return self
    def __floordiv__(self, _o):                return self
    def __rfloordiv__(self, _o):               return self
    def __mod__(self, _o):                     return self
    def __rmod__(self, _o):                    return self
    def __neg__(self):                         return self
    def __pos__(self):                         return self
    def __abs__(self):                         return self


_SENTINEL = _Sentinel()


class _ProbeBinding(dict):
    """Dict-like binding that records every string key accessed."""

    def __init__(self) -> None:
        super().__init__()
        self.accessed: set[str] = set()

    def __getitem__(self, key):
        if isinstance(key, str):
            self.accessed.add(key)
        return _SENTINEL

    def get(self, key, default=None):
        if isinstance(key, str):
            self.accessed.add(key)
        return _SENTINEL

    def __contains__(self, key):
        if isinstance(key, str):
            self.accessed.add(key)
        return True


def _probe_vars(fn: Callable[[Binding], Any] | None) -> set[str]:
    if fn is None:
        return set()
    probe = _ProbeBinding()
    try:
        fn(probe)
    except Exception:
        pass
    return probe.accessed


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def _multiset_to_snk(result: Any) -> Any:
    """Convert a custom ``Multiset`` (Counter) result into a SNAKES value."""
    try:
        items = list(result.items())
    except AttributeError:
        return result
    tokens: list[Any] = []
    for tok, cnt in items:
        tokens.extend([tok] * int(cnt))
    if len(tokens) == 1:
        return tokens[0]
    return SNKMultiSet(tokens)


def translate(net: PetriNet, *, snk_name: str | None = None) -> tuple[SNKNet, list[str]]:
    """Build a SNAKES net equivalent to *net*.

    Returns ``(snakes_net, dropped_transitions)``; ``dropped_transitions``
    names transitions that could not be translated because they reference
    output-only variables (these never fire in the custom engine either).
    """
    snk = SNKNet(snk_name or net.name)
    snk.globals["math"] = math

    def _run_guard(fn, **kwargs):
        return bool(fn(kwargs))

    def _run_arc(fn, **kwargs):
        return _multiset_to_snk(fn(kwargs))

    snk.globals["_run_guard"] = _run_guard
    snk.globals["_run_arc"] = _run_arc

    # Places ----------------------------------------------------------------
    for place in net.places.values():
        tokens: list[Any] = []
        for tok, cnt in place.marking.items():
            tokens.extend([tok] * cnt)
        snk.add_place(SNKPlace(place.name, tokens))

    # Transitions + arcs ----------------------------------------------------
    dropped: list[str] = []
    arc_idx = 0
    guard_idx = 0

    for trans in net.transitions.values():
        in_arcs  = net.input_arcs(trans)
        out_arcs = net.output_arcs(trans)

        in_vars_per_arc  = [_probe_vars(a.expression) for a in in_arcs]
        out_vars_per_arc = [_probe_vars(a.expression) for a in out_arcs]

        input_vars:  set[str] = set().union(*in_vars_per_arc)  if in_vars_per_arc  else set()
        output_vars: set[str] = set().union(*out_vars_per_arc) if out_vars_per_arc else set()
        guard_vars:  set[str] = _probe_vars(trans.guard)

        missing = (output_vars | guard_vars) - input_vars
        if missing:
            dropped.append(trans.name)
            continue

        has_guard = guard_vars and bool(input_vars)
        if has_guard:
            g_key = f"_g_{guard_idx}"; guard_idx += 1
            snk.globals[g_key] = trans.guard
            kw = ", ".join(f"{v}={v}" for v in sorted(input_vars))
            g_expr = Expression(f"_run_guard({g_key}, {kw})")
            snk_trans = SNKTransition(trans.name, g_expr)
        else:
            snk_trans = SNKTransition(trans.name)
        snk.add_transition(snk_trans)

        # SNAKES allows only one arc per (place, transition, direction) triple,
        # so we group arcs by place and fold multiples into a MultiArc.
        kw_sorted = ", ".join(f"{v}={v}" for v in sorted(input_vars))

        def _input_label(arc, vars_):
            nonlocal arc_idx
            if len(vars_) == 1:
                (v,) = vars_
                return Variable(v)
            a_key = f"_a_{arc_idx}"; arc_idx += 1
            snk.globals[a_key] = arc.expression
            return Expression(f"_run_arc({a_key}, {kw_sorted})")

        def _output_label(arc):
            nonlocal arc_idx
            a_key = f"_a_{arc_idx}"; arc_idx += 1
            snk.globals[a_key] = arc.expression
            return Expression(f"_run_arc({a_key}, {kw_sorted})")

        in_by_place: dict[str, list] = {}
        for arc, vars_ in zip(in_arcs, in_vars_per_arc):
            in_by_place.setdefault(arc.place.name, []).append(_input_label(arc, vars_))
        for place_name, labels in in_by_place.items():
            label = labels[0] if len(labels) == 1 else MultiArc(labels)
            snk.add_input(place_name, trans.name, label)

        out_by_place: dict[str, list] = {}
        for arc in out_arcs:
            out_by_place.setdefault(arc.place.name, []).append(_output_label(arc))
        for place_name, labels in out_by_place.items():
            label = labels[0] if len(labels) == 1 else MultiArc(labels)
            snk.add_output(place_name, trans.name, label)

    return snk, dropped
