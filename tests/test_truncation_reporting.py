"""Truncated runs in the subnet, vacuity and timing scripts: a verdict that needs
the whole state space is reported as inconclusive, never as a result."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import run_om_rs_ps_subnets as S
import run_timings as T
import run_vacuity as V


def test_subnet_l5_failure_on_truncated_run_is_inconclusive() -> None:
    assert S._liveness_status(satisfied=False, truncated=True) == "INCONCLUSIVE (truncated)"
    assert S._liveness_status(satisfied=False, truncated=False) == "FAIL"
    assert S._liveness_status(satisfied=True, truncated=True) == "PASS"


def test_subnet_default_limit_covers_ps_n4() -> None:
    assert S.DEFAULT_MAX_STATES >= 112_739      # PS subnet at N = 4


def test_truncated_ps_run_prints_no_l5_fail(capsys: pytest.CaptureFixture[str]) -> None:
    # PS at N = 2 has 766 states; aggregate L5 holds on the full space.
    S._run_ps(2, max_states=100)
    out = capsys.readouterr().out
    assert "Truncated        : True" in out
    assert "L5 (PS_waiting ∧ mem_free → F(PS_has_mem))  INCONCLUSIVE" in out


def test_truncated_vacuity_run_gives_no_verdict(capsys: pytest.CaptureFixture[str]) -> None:
    V.run(2, max_states=100)                    # N = 2 has 581 states
    out = capsys.readouterr().out
    assert "truncated = True" in out
    for word in ("SUBSTANTIVE", "VACUOUS", "STRUCTURAL", "WITNESSED"):
        assert word not in out
    assert out.count("INCONCLUSIVE (truncated)") == 5


def test_timing_run_reports_truncation() -> None:
    assert T._one(2, 4, max_states=100)[3] is True
    assert T._one(2, 4, max_states=1_000)[1:] == (581, 1410, False)
