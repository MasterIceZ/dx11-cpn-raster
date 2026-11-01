
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from pipeline.petri.state_space import FrozenMarking, StateSpace






AP = Callable[[FrozenMarking], bool]





@dataclass(frozen=True)
class Formula:
    pass

@dataclass(frozen=True)
class Atom(Formula):

    prop: AP
    name: str = "<atom>"

    def __repr__(self) -> str:
        return self.name

@dataclass(frozen=True)
class Not(Formula):
    sub: Formula

    def __repr__(self) -> str:
        return f"¬({self.sub})"

@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula

    def __repr__(self) -> str:
        return f"({self.left} ∧ {self.right})"

@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula

    def __repr__(self) -> str:
        return f"({self.left} ∨ {self.right})"

@dataclass(frozen=True)
class Implies(Formula):
    left: Formula
    right: Formula

    def __repr__(self) -> str:
        return f"({self.left} → {self.right})"

@dataclass(frozen=True)
class G(Formula):

    sub: Formula

    def __repr__(self) -> str:
        return f"G({self.sub})"

@dataclass(frozen=True)
class F(Formula):

    sub: Formula

    def __repr__(self) -> str:
        return f"F({self.sub})"

@dataclass(frozen=True)
class X(Formula):

    sub: Formula

    def __repr__(self) -> str:
        return f"X({self.sub})"

@dataclass(frozen=True)
class U(Formula):

    left: Formula
    right: Formula

    def __repr__(self) -> str:
        return f"({self.left} U {self.right})"





@dataclass
class CheckResult:

    formula: Formula
    satisfied: bool
    counterexample: list[FrozenMarking] | None = None
    note: str = ""

    def __repr__(self) -> str:
        status = "SAT ✓" if self.satisfied else "UNSAT ✗"
        cx = f", counterexample len={len(self.counterexample)}" if self.counterexample else ""
        return f"CheckResult({status}{cx}, {self.formula!r})"





def _eval_state(formula: Formula, marking: FrozenMarking) -> bool:
    match formula:
        case Atom(prop=p):
            return p(marking)
        case Not(sub=s):
            return not _eval_state(s, marking)
        case And(left=l, right=r):
            return _eval_state(l, marking) and _eval_state(r, marking)
        case Or(left=l, right=r):
            return _eval_state(l, marking) or _eval_state(r, marking)
        case Implies(left=l, right=r):
            return (not _eval_state(l, marking)) or _eval_state(r, marking)
        case _:
            raise ValueError(
                f"_eval_state: temporal operator "
                f"{type(formula).__name__!r} requires path semantics"
            )

def _path_to(
    ss: StateSpace,
    start: FrozenMarking,
    target: FrozenMarking,
) -> list[FrozenMarking] | None:
    if start == target:
        return [start]
    parent: dict[FrozenMarking, FrozenMarking | None] = {start: None}
    queue: deque[FrozenMarking] = deque([start])
    while queue:
        cur = queue.popleft()
        for _, nxt in ss.successors(cur):
            if nxt == target:
                path: list[FrozenMarking] = [target]
                node: FrozenMarking | None = cur
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return list(reversed(path))
            if nxt not in parent:
                parent[nxt] = cur
                queue.append(nxt)
    return None

def _path_to_pred(
    ss: StateSpace,
    start: FrozenMarking,
    predicate: Callable[[FrozenMarking], bool],
) -> list[FrozenMarking] | None:
    if predicate(start):
        return [start]
    parent: dict[FrozenMarking, FrozenMarking | None] = {start: None}
    queue: deque[FrozenMarking] = deque([start])
    while queue:
        cur = queue.popleft()
        for _, nxt in ss.successors(cur):
            if predicate(nxt):
                path: list[FrozenMarking] = [nxt]
                node: FrozenMarking | None = cur
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return list(reversed(path))
            if nxt not in parent:
                parent[nxt] = cur
                queue.append(nxt)
    return None





def _avoiding_suffix(
    ss: StateSpace,
    start: FrozenMarking,
    good: set[FrozenMarking],
) -> list[FrozenMarking]:
    path: list[FrozenMarking] = [start]
    seen: set[FrozenMarking] = {start}
    cur = start
    while True:
        nxt = next(
            (d for _, d in ss.successors(cur)
             if d not in good and d in ss.states and d != cur),
            None,
        )
        if nxt is None:
            return path
        path.append(nxt)
        if nxt in seen:
            return path
        seen.add(nxt)
        cur = nxt





class LTLChecker:

    def check(self, ss: StateSpace, formula: Formula) -> CheckResult:

        match formula:
            case G(sub=Implies(left=p, right=F(sub=q))):
                return self._check_response(ss, p, q, formula)
            case G(sub=F(sub=sub)):
                return self._check_finally(ss, sub, formula)
            case G(sub=sub):
                return self._check_globally(ss, sub, formula)
            case F(sub=sub):
                return self._check_finally(ss, sub, formula)
            case _:
                return self._check_state_formula(ss, formula)





    def _check_globally(
        self, ss: StateSpace, phi: Formula, outer: Formula
    ) -> CheckResult:
        for m in ss.states:
            try:
                if not _eval_state(phi, m):
                    path = _path_to(ss, ss.initial, m)
                    return CheckResult(outer, False, counterexample=path or [m])
            except ValueError as exc:
                return CheckResult(outer, False, note=str(exc))
        return CheckResult(outer, True)





    def _check_finally(
        self, ss: StateSpace, phi: Formula, outer: Formula
    ) -> CheckResult:
        def sat(m: FrozenMarking) -> bool:
            try:
                return _eval_state(phi, m)
            except ValueError:
                return False

        sccs = ss.sccs()


        state_to_scc: dict[FrozenMarking, int] = {}
        for i, scc in enumerate(sccs):
            for s in scc:
                state_to_scc[s] = i


        scc_has_exit: list[bool] = [False] * len(sccs)
        for src in ss.states:
            si = state_to_scc[src]
            for _, dst in ss.successors(src):
                if state_to_scc[dst] != si:
                    scc_has_exit[si] = True

        for i, scc in enumerate(sccs):
            if scc_has_exit[i]:
                continue
            if any(sat(m) for m in scc):
                continue

            witness = next(iter(scc))
            path = _path_to_pred(ss, ss.initial, lambda s: s in scc)
            return CheckResult(outer, False, counterexample=path or [witness])

        return CheckResult(outer, True)





    def _check_response(
        self,
        ss: StateSpace,
        p: Formula,
        q: Formula,
        outer: Formula,
    ) -> CheckResult:
        def sat_p(m: FrozenMarking) -> bool:
            try:
                return _eval_state(p, m)
            except ValueError:
                return False

        def sat_q(m: FrozenMarking) -> bool:
            try:
                return _eval_state(q, m)
            except ValueError:
                return False

        if ss.truncated:
            return CheckResult(
                outer, False,
                note="inconclusive: state space truncated, F(q) undecidable at frontier",
            )

        reverse: dict[FrozenMarking, list[FrozenMarking]] = {
            m: [] for m in ss.states
        }
        pending: dict[FrozenMarking, set[FrozenMarking]] = {}
        for src in ss.states:
            succs = {
                dst for _, dst in ss.successors(src)
                if dst in reverse and dst != src
            }
            pending[src] = succs
            for dst in succs:
                reverse[dst].append(src)

        # AF(q): least fixpoint of the q-states plus every state whose *every*
        # successor is already inside.  Self-loop edges (net-zero firings such as
        # OM_depth_test) are stuttering steps and ignored, as _check_finally
        # already does for GF -- i.e. weak fairness.  A state with no other
        # successor is terminal, so a terminal ¬q-state stays out.  Backward
        # reachability alone would decide only EF(q), not F(q) on all paths.
        af_q: set[FrozenMarking] = set()
        queue: deque[FrozenMarking] = deque()
        for m in ss.states:
            if sat_q(m):
                af_q.add(m)
                queue.append(m)
        while queue:
            cur = queue.popleft()
            for pred in reverse[cur]:
                if pred in af_q:
                    continue
                pending[pred].discard(cur)
                if not pending[pred]:
                    af_q.add(pred)
                    queue.append(pred)

        for m in ss.states:
            if sat_p(m) and m not in af_q:
                prefix = _path_to(ss, ss.initial, m) or [m]
                return CheckResult(
                    outer, False,
                    counterexample=prefix + _avoiding_suffix(ss, m, af_q)[1:],
                )

        return CheckResult(outer, True)





    def _check_state_formula(
        self, ss: StateSpace, formula: Formula
    ) -> CheckResult:
        for m in ss.states:
            try:
                if not _eval_state(formula, m):
                    path = _path_to(ss, ss.initial, m)
                    return CheckResult(formula, False, counterexample=path or [m])
            except ValueError as exc:
                return CheckResult(
                    formula,
                    False,
                    note=f"Unsupported formula pattern: {exc}",
                )
        return CheckResult(formula, True)
