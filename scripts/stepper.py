"""
Text-mode interactive stepper for the DX11 CPN pipeline.

Usage:
    uv run python scripts/stepper.py          # 1 vertex, interactive
    uv run python scripts/stepper.py --n 2    # 2 vertices
    uv run python scripts/stepper.py --auto   # auto-step (no input needed)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.petri.colored import Vertex
from pipeline.petri.net import PetriNet
from pipeline.petri.state_space import freeze, thaw
from models.dx11_full_pipeline import build_full_pipeline
from scripts.run_full_pipeline import _full_pipeline_fire_all

# ── places to hide (static resource bindings that never change) ──────────────
_STATIC_PREFIXES = (
    "VS_lane", "PS_lane",
    "VS_shader", "VS_cbuffer",
    "HS_shader", "HS_cbuffer", "HS_tess_factor", "HS_max_tess_factor",
    "HS_domain_type", "HS_sampler", "HS_bypass",
    "DS_shader", "DS_cbuffer", "DS_sampler", "DS_srv", "DS_bypass",
    "GS_shader", "GS_cbuffer", "GS_sampler", "GS_stream_select",
    "GS_bypass", "GS_max_output_count", "GS_instance_count",
    "GS_output_topology", "GS_cut_buffer",
    "SO_bypass", "SO_write_offset",
    "RS_rasterizer_state", "RS_viewport", "RS_scissor",
    "RS_coverage_mask", "RS_fill_mode", "RS_cull_mode",
    "RS_depth_clip_enable", "RS_msaa_state", "RS_line_aa_state",
    "RS_conservative_raster",
    "PS_shader", "PS_cbuffer", "PS_sampler", "PS_srv",
    "OM_blend_state", "OM_depth_stencil_state", "OM_dsv", "OM_rt_slot",
    "Tess_factor_buffer", "Tess_topology_out", "Tess_partitioning",
    "Tess_wind_order", "Tess_bypass",
    "IA_input_layout", "IA_topology", "IA_draw_cmd", "IA_patch_size",
    "IA_vertex_id", "IA_indirect_args",
)

# ── stage a place belongs to (for grouping) ──────────────────────────────────
_STAGE_ORDER = ["IA", "VS", "HS", "Tess", "DS", "GS", "SO", "RS", "PS", "OM"]

def _stage(place: str) -> str:
    for s in _STAGE_ORDER:
        if place.startswith(s + "_"):
            return s
    return "?"

def _is_static(place: str) -> bool:
    return any(place.startswith(p) for p in _STATIC_PREFIXES)

def _token_label(tok) -> str:
    cls = type(tok).__name__
    if hasattr(tok, "vertex_id"):
        return f"Vertex(id={tok.vertex_id})"
    if hasattr(tok, "topology"):
        return f"Primitive({tok.topology}, {len(tok.vertices)}v)"
    if hasattr(tok, "pixel_x"):
        if hasattr(tok, "color"):
            return f"Pixel({tok.pixel_x},{tok.pixel_y} d={tok.depth:.2f})"
        return f"Fragment({tok.pixel_x},{tok.pixel_y} d={tok.depth:.2f})"
    if hasattr(tok, "patch_id"):
        return f"Patch(id={tok.patch_id}, {len(tok.vertices)}v)"
    if hasattr(tok, "u") and hasattr(tok, "v"):
        return f"DomainPoint(u={tok.u:.2f},v={tok.v:.2f})"
    if hasattr(tok, "lane"):
        return f"LaneSlot({tok.lane})"
    if hasattr(tok, "slot") and hasattr(tok, "size"):
        return f"CBuffer(slot={tok.slot})"
    if hasattr(tok, "stage") and hasattr(tok, "name"):
        return f"Shader({tok.stage}/{tok.name})"
    if hasattr(tok, "stream_index"):
        return f"StreamRecord(s={tok.stream_index})"
    return repr(tok)

def _effective_firings(net: PetriNet) -> list[tuple[str, dict]]:
    """Return enabled firings that actually change the marking (filter self-loops)."""
    before = freeze(net)
    result = []
    for tname, binding in _full_pipeline_fire_all(net):
        net.fire(tname, binding)
        after = freeze(net)
        thaw(net, before)
        if after != before:
            result.append((tname, binding))
    return result


def _print_state(net: PetriNet, step: int) -> None:
    print(f"\n{'─'*60}")
    print(f"  Step {step}  —  pipeline state")
    print(f"{'─'*60}")

    by_stage: dict[str, list[str]] = {s: [] for s in _STAGE_ORDER + ["?"]}
    for name, place in net.places.items():
        if _is_static(name):
            continue
        tokens = list(place.marking)
        if not tokens:
            continue
        stage = _stage(name)
        for tok in tokens:
            by_stage[stage].append(f"  {name:35s} ← {_token_label(tok)}")

    any_shown = False
    for stage in _STAGE_ORDER:
        lines = by_stage[stage]
        if lines:
            print(f"  [{stage}]")
            for l in lines:
                print(l)
            any_shown = True

    if not any_shown:
        print("  (pipeline empty — all tokens in sink places)")

    # show sink totals
    sinks = {
        f"OM_written_pixel_{i}": len(list(net.places[f"OM_written_pixel_{i}"].marking))
        for i in range(8)
        if f"OM_written_pixel_{i}" in net.places
    }
    written = sum(sinks.values())
    if written:
        print(f"\n  OM sinks: {written} pixel(s) written to render target")
    if "OM_rt_active" in net.places and list(net.places["OM_rt_active"].marking):
        print("  OM_rt_active: UNLOCKED ✓")
    elif "OM_rt_in_use" in net.places and list(net.places["OM_rt_in_use"].marking):
        print("  OM_rt_active: LOCKED (write in progress)")

def _print_firings(firings: list[tuple[str, dict]]) -> None:
    print(f"\n  Available transitions ({len(firings)}):")
    for i, (tname, _) in enumerate(firings):
        print(f"    [{i}] {tname}")

def _prompt(firings: list, auto: bool) -> int | None:
    if not firings:
        return None
    if auto:
        return 0           # always pick first (deterministic demo)
    while True:
        raw = input("\n  pick [0-{}], Enter=0, a=auto, q=quit: ".format(len(firings)-1)).strip()
        if raw in ("", "0") or raw == "0":
            return 0
        if raw == "q":
            return None
        if raw == "a":
            return -1      # signal: switch to auto
        try:
            n = int(raw)
            if 0 <= n < len(firings):
                return n
        except ValueError:
            pass
        print("  invalid choice")

def run(n_vertices: int, auto: bool) -> None:
    net = build_full_pipeline()
    for i in range(n_vertices):
        v = Vertex(
            vertex_id=i,
            clip_pos=(float(i + 1), float(i), 0.5, 1.0),
            instance_id=0,
            attributes=(),
        )
        net.places["IA_vertex_buffer"].add_token(v)

    print(f"\n{'='*60}")
    print(f"  DX11 Pipeline Stepper  |  N = {n_vertices} vertex/vertices")
    print(f"  mode: {'auto' if auto else 'interactive'}")
    print(f"{'='*60}")

    step = 0
    _print_state(net, step)

    while True:
        firings = _effective_firings(net)

        if not firings:
            print(f"\n  ✓ Pipeline drained — no more transitions enabled.")
            print(f"  Total steps: {step}")
            break

        if not auto:
            _print_firings(firings)

        choice = _prompt(firings, auto)

        if choice is None:
            print("\n  Stopped.")
            break
        if choice == -1:
            auto = True
            choice = 0

        tname, binding = firings[choice]
        net.fire(tname, binding)
        step += 1

        if auto:
            print(f"  [{step:3d}] fired: {tname}")
        else:
            print(f"\n  → fired: {tname}")
            _print_state(net, step)

    if auto:
        _print_state(net, step)

def main() -> None:
    parser = argparse.ArgumentParser(description="DX11 CPN text-mode stepper")
    parser.add_argument("--n", type=int, default=1, help="number of input vertices")
    parser.add_argument("--auto", action="store_true", help="auto-step without prompts")
    args = parser.parse_args()
    run(args.n, args.auto)

if __name__ == "__main__":
    main()
