
from __future__ import annotations

from typing import Any

from pipeline.petri.colored import (
    Binding,
    Fragment,
    Primitive,
    guard,
)
from pipeline.petri.net import Multiset, PetriNet
from pipeline.stages.vertex_shader import ShaderBinding
from models.dx11_full_pipeline import ScissorRect, ViewportRect


def _p(var: str):
    return lambda b, _v=var: Multiset.of(b[_v])


def _ro(net: PetriNet, place: str, transition: str, var: str) -> None:
    net.add_input_arc(place, transition, _p(var))
    net.add_output_arc(place, transition, _p(var))


def _sb(name: str = "default", valid: bool = True) -> ShaderBinding:
    return ShaderBinding(stage="RS", name=name, valid=valid)


@guard("rs_rasterize_triangle_guard")
def rs_rasterize_triangle_guard(b: Binding) -> bool:
    prim: Primitive | None = b.get("prim")
    return prim is not None and prim.topology == "triangle_list"


@guard("rs_rasterize_line_guard")
def rs_rasterize_line_guard(b: Binding) -> bool:
    prim: Primitive | None = b.get("prim")
    return prim is not None and prim.topology == "line_list"


@guard("rs_rasterize_point_guard")
def rs_rasterize_point_guard(b: Binding) -> bool:
    prim: Primitive | None = b.get("prim")
    return prim is not None and prim.topology not in ("triangle_list", "line_list")


def build_subnet(config: dict[str, Any] | None = None) -> PetriNet:
    """Standalone Rasterizer (RS) subnet.

    ``RS_input`` is the stage's input interface (clipped/assembled primitives
    arriving from GS/SO, exactly as in the composed pipeline) and
    ``PS_output``'s sibling, ``PS_input``, is wired in here as RS's *output*
    interface — RS deposits rasterized ``Fragment`` tokens there and the four
    PS lanes drain them. ``PS_input`` is precisely the place the corrected L4
    atomic proposition (``_rs_queue_within_bound`` in
    ``pipeline.verifier.dx11_ltl``) inspects, so checking L4 against this
    subnet is a genuine, non-vacuous rasterizer-queue-boundedness check
    rather than the vacuous VS-subnet pass the paper originally (and
    incorrectly) reported.

    Clip-plane and viewport/scissor fan-out is reduced from the composed
    model's 6 planes / 2 viewports / 2 scissor rects to one of each — every
    ``_clip_plane`` instance is functionally identical (``RS_input →
    RS_clipped_primitive`` gated only by ``RS_depth_clip_enable``), so the
    extra copies contribute interleavings, not new behaviour, and only
    inflate the state space. This mirrors the same single-instance
    simplification ``vertex_shader.build_subnet`` already applies to VS's
    four-lane structure.
    """
    cfg: dict[str, Any] = config or {}
    max_primitives: int = cfg.get("max_primitives", 8_192)
    max_fragments: int = cfg.get("max_fragments", 65_536)
    W = int(cfg.get("render_width", 64))
    H = int(cfg.get("render_height", 64))

    net = PetriNet(name="RS")

    net.add_place("RS_input", Primitive, bound=max_primitives)
    net.add_place("RS_clipped_primitive", Primitive, bound=max_primitives)
    net.add_place("RS_culled_primitive", Primitive, bound=max_primitives)
    net.add_place("PS_input", Fragment, bound=max_fragments)

    net.add_place("RS_depth_clip_enable", ShaderBinding, bound=1,
                  initial=[_sb("enabled")])
    net.add_place("RS_cull_mode", ShaderBinding, bound=1,
                  initial=[_sb("back")])
    net.add_place("RS_rasterizer_state", ShaderBinding, bound=1,
                  initial=[_sb("Default")])
    net.add_place("RS_coverage_mask", ShaderBinding, bound=1,
                  initial=[_sb("0xFFFF")])
    net.add_place("RS_fill_mode", ShaderBinding, bound=1,
                  initial=[_sb("solid")])
    net.add_place("RS_msaa_state", ShaderBinding, bound=1,
                  initial=[_sb("1x")])
    net.add_place("RS_line_aa_state", ShaderBinding, bound=1,
                  initial=[_sb("disabled")])
    net.add_place("RS_conservative_raster", ShaderBinding, bound=1,
                  initial=[_sb("disabled")])
    net.add_place("RS_viewport_0", ViewportRect, bound=1,
                  initial=[ViewportRect(slot=0, x=0.0, y=0.0, width=W, height=H)])
    net.add_place("RS_scissor_0", ScissorRect, bound=1,
                  initial=[ScissorRect(slot=0, left=0, top=0, right=W, bottom=H)])

    net.add_transition("RS_clip_near")
    _ro(net, "RS_depth_clip_enable", "RS_clip_near", "dce")
    net.add_input_arc("RS_input", "RS_clip_near", _p("prim"))
    net.add_output_arc("RS_clipped_primitive", "RS_clip_near", _p("prim"))

    net.add_transition("RS_clip_far")
    _ro(net, "RS_depth_clip_enable", "RS_clip_far", "dce")
    net.add_input_arc("RS_input", "RS_clip_far", _p("prim"))
    net.add_output_arc("RS_clipped_primitive", "RS_clip_far", _p("prim"))

    for cull in ("RS_cull_back", "RS_cull_front"):
        net.add_transition(cull)
        net.add_input_arc("RS_clipped_primitive", cull, _p("prim"))
        _ro(net, "RS_cull_mode", cull, "cmode")
        net.add_output_arc("RS_culled_primitive", cull, _p("prim"))

    net.add_transition("RS_viewport_transform_0")
    net.add_input_arc("RS_clipped_primitive", "RS_viewport_transform_0", _p("prim"))
    _ro(net, "RS_viewport_0", "RS_viewport_transform_0", "vp")
    net.add_output_arc("RS_clipped_primitive", "RS_viewport_transform_0", _p("prim"))

    net.add_transition("RS_scissor_test_0")
    net.add_input_arc("RS_clipped_primitive", "RS_scissor_test_0", _p("prim"))
    _ro(net, "RS_scissor_0", "RS_scissor_test_0", "sc")
    net.add_output_arc("RS_clipped_primitive", "RS_scissor_test_0", _p("prim"))

    net.add_transition("RS_check_msaa")
    net.add_input_arc("RS_clipped_primitive", "RS_check_msaa", _p("prim"))
    _ro(net, "RS_msaa_state", "RS_check_msaa", "msaa")
    net.add_output_arc("RS_clipped_primitive", "RS_check_msaa", _p("prim"))

    net.add_transition("RS_check_conservative_raster")
    net.add_input_arc("RS_clipped_primitive", "RS_check_conservative_raster", _p("prim"))
    _ro(net, "RS_conservative_raster", "RS_check_conservative_raster", "cr")
    net.add_output_arc("RS_clipped_primitive", "RS_check_conservative_raster", _p("prim"))

    net.add_transition("RS_rasterize_triangle", guard=rs_rasterize_triangle_guard)
    net.add_input_arc("RS_clipped_primitive", "RS_rasterize_triangle", _p("prim"))
    _ro(net, "RS_rasterizer_state", "RS_rasterize_triangle", "rs")
    _ro(net, "RS_coverage_mask", "RS_rasterize_triangle", "cov")
    _ro(net, "RS_fill_mode", "RS_rasterize_triangle", "fill")
    net.add_output_arc(
        "PS_input", "RS_rasterize_triangle",
        lambda b: Multiset.of(
            Fragment(pixel_x=int(b["prim"].vertices[0].clip_pos[0]),
                     pixel_y=int(b["prim"].vertices[0].clip_pos[1]),
                     depth=b["prim"].vertices[0].clip_pos[2])))

    net.add_transition("RS_rasterize_line", guard=rs_rasterize_line_guard)
    net.add_input_arc("RS_clipped_primitive", "RS_rasterize_line", _p("prim"))
    _ro(net, "RS_line_aa_state", "RS_rasterize_line", "laa")
    net.add_output_arc(
        "PS_input", "RS_rasterize_line",
        lambda b: Multiset.of(Fragment(pixel_x=0, pixel_y=0, depth=0.5)))

    net.add_transition("RS_rasterize_point", guard=rs_rasterize_point_guard)
    net.add_input_arc("RS_clipped_primitive", "RS_rasterize_point", _p("prim"))
    net.add_output_arc(
        "PS_input", "RS_rasterize_point",
        lambda b: Multiset.of(Fragment(pixel_x=0, pixel_y=0, depth=0.5)))

    net.add_transition("RS_interpolate_attributes")
    net.add_input_arc("PS_input", "RS_interpolate_attributes", _p("frag"))
    net.add_output_arc("PS_input", "RS_interpolate_attributes", _p("frag"))

    return net
