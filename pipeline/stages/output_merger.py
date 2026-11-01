from __future__ import annotations

from typing import Any

from pipeline.petri.colored import (
    Binding,
    Pixel,
    RenderTarget,
    guard,
)
from pipeline.petri.net import Multiset, PetriNet
from pipeline.stages.vertex_shader import ShaderBinding
from models.dx11_full_pipeline import BlendState, DepthStencilState


N_RT_SLOTS = 2


def _p(var: str):
    return lambda b, _v=var: Multiset.of(b[_v])


def _ro(net: PetriNet, place: str, transition: str, var: str) -> None:
    net.add_input_arc(place, transition, _p(var))
    net.add_output_arc(place, transition, _p(var))


@guard("om_depth_test_less")
def om_depth_less(b: Binding) -> bool:
    px: Pixel | None = b.get("px")
    dss: DepthStencilState | None = b.get("dss")
    return (px is not None and dss is not None
            and dss.depth_enable and dss.depth_func == "LESS"
            and px.depth < 1.0)


@guard("om_pixel_in_rt")
def om_pixel_in_rt(b: Binding) -> bool:
    px: Pixel | None = b.get("px")
    rt: RenderTarget | None = b.get("rt")
    return (px is not None and rt is not None
            and 0 <= px.pixel_x < rt.width
            and 0 <= px.pixel_y < rt.height
            and px.output_slot == rt.slot)


def build_subnet(config: dict[str, Any] | None = None) -> PetriNet:
    """Standalone Output Merger (OM) subnet.

    ``PS_output`` is wired in as the OM's *input interface* — exactly as it
    is in the composed pipeline (``models.dx11_full_pipeline``), where pixels
    coming out of the four PS lanes accumulate here and are then read,
    tested, blended and written back in place by the OM transitions. This is
    the same place the corrected L2 atomic proposition
    (``_concurrent_rt_write`` in ``pipeline.verifier.dx11_ltl``) inspects,
    so checking L2 against this subnet is a genuine, non-vacuous OM-mutual-
    exclusion check rather than the vacuous VS-subnet pass the paper
    originally (and incorrectly) reported.

    Render-target fan-out is reduced from the composed model's 8 slots to
    ``N_RT_SLOTS`` (2) to keep the reachable state space tractable for BFS —
    two slots are already enough for two independently-addressable pixels to
    race for the same target, which is what L2 guards against.
    """
    cfg: dict[str, Any] = config or {}
    max_pixels: int = cfg.get("max_pixels", 65_536)
    W = int(cfg.get("render_width", 64))
    H = int(cfg.get("render_height", 64))

    net = PetriNet(name="OM")

    net.add_place("PS_output", Pixel, bound=max_pixels)

    net.add_place("OM_blend_state", BlendState, bound=1,
                  initial=[BlendState(blend_enable=False)])
    net.add_place("OM_depth_stencil_state", DepthStencilState, bound=1,
                  initial=[DepthStencilState(depth_enable=True,
                                             depth_write_mask=True,
                                             depth_func="LESS")])
    for i in range(N_RT_SLOTS):
        net.add_place(f"OM_rt_slot_{i}", RenderTarget, bound=1,
                      initial=[RenderTarget(slot=i, width=W, height=H)])
    net.add_place("OM_dsv", RenderTarget, bound=1,
                  initial=[RenderTarget(slot=-1, width=W, height=H)])

    for i in range(N_RT_SLOTS):
        net.add_place(f"OM_written_pixel_{i}", Pixel, bound=None)
    net.add_place("OM_depth_buffer", Pixel, bound=None)
    net.add_place("OM_stencil_buffer", Pixel, bound=None)

    net.add_place("OM_rt_active", ShaderBinding, bound=1,
                  initial=[ShaderBinding(stage="OM", name="rt_active", valid=True)])
    net.add_place("OM_rt_in_use", ShaderBinding, bound=1)

    net.add_transition("OM_depth_test", guard=om_depth_less)
    net.add_input_arc("PS_output", "OM_depth_test", _p("px"))
    _ro(net, "OM_depth_stencil_state", "OM_depth_test", "dss")
    _ro(net, "OM_dsv", "OM_depth_test", "rt")
    net.add_output_arc("PS_output", "OM_depth_test", _p("px"))

    net.add_transition("OM_stencil_test")
    net.add_input_arc("PS_output", "OM_stencil_test", _p("px"))
    _ro(net, "OM_depth_stencil_state", "OM_stencil_test", "dss")
    net.add_output_arc("PS_output", "OM_stencil_test", _p("px"))

    net.add_transition("OM_depth_write")
    net.add_input_arc("PS_output", "OM_depth_write", _p("px"))
    _ro(net, "OM_depth_stencil_state", "OM_depth_write", "dss")
    _ro(net, "OM_dsv", "OM_depth_write", "rt")
    net.add_output_arc("PS_output", "OM_depth_write", _p("px"))
    net.add_output_arc("OM_depth_buffer", "OM_depth_write", _p("px"))

    net.add_transition("OM_stencil_write")
    net.add_input_arc("PS_output", "OM_stencil_write", _p("px"))
    _ro(net, "OM_depth_stencil_state", "OM_stencil_write", "dss")
    net.add_output_arc("OM_stencil_buffer", "OM_stencil_write", _p("px"))
    net.add_output_arc("PS_output", "OM_stencil_write", _p("px"))

    for i in range(N_RT_SLOTS):
        net.add_transition(f"OM_blend_rt_{i}", guard=om_pixel_in_rt)
        net.add_input_arc("PS_output", f"OM_blend_rt_{i}", _p("px"))
        _ro(net, "OM_blend_state", f"OM_blend_rt_{i}", "bs")
        _ro(net, f"OM_rt_slot_{i}", f"OM_blend_rt_{i}", "rt")
        net.add_output_arc("PS_output", f"OM_blend_rt_{i}", _p("px"))

    for i in range(N_RT_SLOTS):
        net.add_transition(f"OM_write_rt_{i}", guard=om_pixel_in_rt)
        net.add_input_arc("PS_output", f"OM_write_rt_{i}", _p("px"))
        net.add_input_arc("OM_rt_active", f"OM_write_rt_{i}", _p("ra"))
        _ro(net, f"OM_rt_slot_{i}", f"OM_write_rt_{i}", "rt")
        _ro(net, "OM_blend_state", f"OM_write_rt_{i}", "bs")
        net.add_output_arc(f"OM_written_pixel_{i}", f"OM_write_rt_{i}", _p("px"))
        net.add_output_arc("OM_rt_in_use", f"OM_write_rt_{i}", _p("ra"))

    net.add_transition("OM_release_rt")
    net.add_input_arc("OM_rt_in_use", "OM_release_rt", _p("ra"))
    net.add_output_arc("OM_rt_active", "OM_release_rt", _p("ra"))

    net.add_transition("OM_discard_depth_fail")
    net.add_input_arc("PS_output", "OM_discard_depth_fail", _p("px"))
    _ro(net, "OM_depth_stencil_state", "OM_discard_depth_fail", "dss")

    net.add_transition("OM_discard_stencil_fail")
    net.add_input_arc("PS_output", "OM_discard_stencil_fail", _p("px"))
    _ro(net, "OM_depth_stencil_state", "OM_discard_stencil_fail", "dss")

    return net
