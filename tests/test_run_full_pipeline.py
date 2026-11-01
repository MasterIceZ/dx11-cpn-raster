"""Reporting in scripts/run_full_pipeline.py: truncated runs and the default
state limit (the plain command must reproduce Table III's four-lane rows)."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import run_full_pipeline as R


def test_truncated_run_reports_inconclusive_not_fail(capsys: pytest.CaptureFixture[str]) -> None:
    # N = 2 has 581 states, so a 100-state limit truncates it.  Every property
    # holds on the full space; a frontier state merely looks dead.
    R.run(2, max_states=100)
    out = capsys.readouterr().out
    assert "Truncated         : True" in out
    assert "FAIL" not in out
    assert "INCONCLUSIVE" in out


def test_invariant_violation_on_truncated_run_is_still_a_fail() -> None:
    # A state violating L2 or L4 is reachable whether or not BFS finished.
    assert R.status("P2", satisfied=False, truncated=True) == "✗  FAIL"
    assert R.status("P4", satisfied=False, truncated=True) == "✗  FAIL"
    assert R.status("P6", satisfied=False, truncated=True).startswith("?  INCONCLUSIVE")
    assert R.status("P6", satisfied=False, truncated=False) == "✗  FAIL"
    assert R.status("P6", satisfied=True, truncated=True) == "✓  PASS"


def test_default_state_limit_does_not_truncate_n4(monkeypatch: pytest.MonkeyPatch) -> None:
    run_default = inspect.signature(R.run).parameters["max_states"].default
    seen: dict[str, int] = {}
    monkeypatch.setattr(R, "run", lambda n, **kw: seen.setdefault("max_states", kw["max_states"]))
    monkeypatch.setattr(sys, "argv", ["run_full_pipeline.py", "--n", "4"])
    R.main()
    assert seen["max_states"] >= 208_905      # four-lane N = 4, Table III
    assert run_default == seen["max_states"]
