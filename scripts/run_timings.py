"""Repeat the full-pipeline BFS for each reported configuration and report the
median wall-clock time of three runs, so Table III's timings are not single-run
measurements.  State and edge counts are deterministic and are reported once.

Usage:
    uv run python scripts/run_timings.py --repeats 3
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import platform
import statistics
import time

from models.dx11_full_pipeline import build_full_pipeline
from pipeline.petri.colored import Vertex
from pipeline.petri.state_space import build_state_space

from run_full_pipeline import LARGE_MAX_STATES, _full_pipeline_fire_all

CONFIGS: tuple[tuple[int, int], ...] = (
    (1, 4), (2, 4), (3, 4), (4, 4), (3, 2), (4, 2),
    (4, 3), (5, 3), (5, 2), (4, 1), (5, 1), (6, 1),
)


def _one(n: int, lanes: int, max_states: int):
    net = build_full_pipeline({"lanes": lanes})
    for i in range(n):
        net.places["IA_vertex_buffer"].add_token(
            Vertex(vertex_id=i, clip_pos=(float(i + 1), float(i), 0.5, 1.0),
                   instance_id=0, attributes=())
        )
    t0 = time.perf_counter()
    ss = build_state_space(net, _full_pipeline_fire_all, max_states=max_states)
    return (time.perf_counter() - t0, len(ss.states),
            sum(len(v) for v in ss.edges.values()), ss.truncated)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-states", type=int, default=LARGE_MAX_STATES)
    ap.add_argument(
        "--configs", nargs="+", metavar="N:LANES", default=None,
        help="Time only these configurations, e.g. 5:3 6:1 (default: all of CONFIGS)",
    )
    args = ap.parse_args()
    configs = (
        [tuple(int(x) for x in c.split(":")) for c in args.configs]
        if args.configs else CONFIGS
    )

    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(f"{'N':>3} {'lanes':>6} {'states':>8} {'edges':>9} "
          f"{'median(s)':>10} {'min':>8} {'max':>8}")
    for n, lanes in configs:
        times, states, edges = [], None, None
        for _ in range(args.repeats):
            t, s, e, truncated = _one(n, lanes, args.max_states)
            times.append(t)
            if states is not None and (s, e) != (states, edges):
                raise AssertionError(f"non-deterministic exploration at N={n}, lanes={lanes}")
            states, edges = s, e
        print(f"{n:>3} {lanes:>6} {states:>8} {edges:>9} "
              f"{statistics.median(times):>10.2f} {min(times):>8.2f} {max(times):>8.2f}"
              + ("  TRUNCATED at --max-states: not the full state space" if truncated else ""),
              flush=True)


if __name__ == "__main__":
    main()
