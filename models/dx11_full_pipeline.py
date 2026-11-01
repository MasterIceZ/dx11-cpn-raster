
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from pipeline.petri.colored import (
    Binding, Fragment, Pixel, Primitive, RenderTarget, Vertex, guard, GuardFunction,
)
from pipeline.petri.net import Multiset, PetriNet
from pipeline.stages.vertex_shader import ConstantBuffer, ShaderBinding

@dataclass(frozen=True)
class LaneSlot:
    lane: int

@dataclass(frozen=True)
class Index:
    value: int
    instance_id: int = 0

@dataclass(frozen=True)
class Patch:
    vertices: tuple[Vertex, ...]
    patch_id: int = 0
    instance_id: int = 0

@dataclass(frozen=True)
class DomainPoint:
    patch: Patch
    u: float
    v: float
    w: float = 0.0
    domain_id: int = 0

@dataclass(frozen=True)
class SamplerState:
    slot: int
    filter: str = "LINEAR"
    address_u: str = "CLAMP"
    address_v: str = "CLAMP"

@dataclass(frozen=True)
class BlendState:
    blend_enable: bool = False
    src_blend: str = "ONE"
    dst_blend: str = "ZERO"
    blend_op: str = "ADD"
    rt_write_mask: int = 0xFF

@dataclass(frozen=True)
class DepthStencilState:
    depth_enable: bool = True
    depth_write_mask: bool = True
    depth_func: str = "LESS"
    stencil_enable: bool = False
    stencil_read_mask: int = 0xFF
    stencil_write_mask: int = 0xFF

@dataclass(frozen=True)
class ViewportRect:
    slot: int
    x: float
    y: float
    width: float
    height: float
    min_depth: float = 0.0
    max_depth: float = 1.0

@dataclass(frozen=True)
class ScissorRect:
    slot: int
    left: int
    top: int
    right: int
    bottom: int

@dataclass(frozen=True)
class StreamRecord:
    stream_index: int
    primitive: Primitive
    byte_offset: int = 0


def _p(var: str):
    return lambda b, _v=var: Multiset.of(b[_v])

def _ro(net: PetriNet, place: str, transition: str, var: str) -> None:
    net.add_input_arc(place, transition, _p(var))
    net.add_output_arc(place, transition, _p(var))

def _sb(stage: str, name: str = "default", valid: bool = True) -> ShaderBinding:
    return ShaderBinding(stage=stage, name=name, valid=valid)


def _vs_transform(v: Vertex, _cbuf: ConstantBuffer) -> Vertex:
    x, y, z, w = v.clip_pos
    return Vertex(
        vertex_id=v.vertex_id,
        clip_pos=(x * 0.5, y, z, w),
        instance_id=v.instance_id,
        attributes=v.attributes,
    )


def _ps_transform(frag: Fragment) -> Pixel:
    return Pixel(
        pixel_x=frag.pixel_x,
        pixel_y=frag.pixel_y,
        color=(1.0, 0.0, 0.0, 1.0),
        depth=frag.depth,
        output_slot=0,
    )


@guard("vs_exec_guard")
def _vs_exec_guard(b: Binding) -> bool:
    s = b.get("vs_shader"); c = b.get("cbuf"); v = b.get("v")
    if not (s and c and v):
        return False
    if not (s.valid and 0 <= c.slot <= 13):
        return False
    v_out = _vs_transform(v, c)
    x, y, z, w = v_out.clip_pos
    return all(math.isfinite(f) for f in (x, y, z, w)) and w != 0.0

@guard("ps_exec_guard")
def _ps_exec_guard(b: Binding) -> bool:
    s = b.get("ps_shader"); c = b.get("cbuf"); frag = b.get("frag")
    if not (s and c and frag):
        return False
    if not (s.valid and 0 <= c.slot <= 13 and 0.0 <= frag.depth <= 1.0):
        return False
    px = _ps_transform(frag)
    return 0 <= px.output_slot <= 7

@guard("hs_exec_guard")
def _hs_exec_guard(b: Binding) -> bool:
    s = b.get("hs_shader"); c = b.get("cbuf")
    return bool(s and s.valid and c and 0 <= c.slot <= 13)

@guard("ds_exec_guard")
def _ds_exec_guard(b: Binding) -> bool:
    s = b.get("ds_shader"); c = b.get("cbuf")
    v = b.get("v_out");     dp = b.get("dp")
    if not (s and c and v and dp):
        return False
    x, y, z, w = v.clip_pos
    return (s.valid and 0 <= c.slot <= 13
            and all(math.isfinite(f) for f in (x, y, z, w)) and w != 0.0
            and 0.0 <= dp.u <= 1.0 and 0.0 <= dp.v <= 1.0)

@guard("tess_factor_valid")
def _tess_factor_valid(b: Binding) -> bool:
    tf: ConstantBuffer | None = b.get("tf")
    return (tf is not None and len(tf.data) > 0
            and all(0.0 <= f <= 64.0 for f in tf.data))

@guard("om_depth_test_less")
def _om_depth_less(b: Binding) -> bool:
    px: Pixel | None = b.get("px")
    dss: DepthStencilState | None = b.get("dss")
    return (px is not None and dss is not None
            and dss.depth_enable and dss.depth_func == "LESS"
            and px.depth < 1.0)

@guard("om_pixel_in_rt")
def _om_pixel_in_rt(b: Binding) -> bool:
    px: Pixel | None = b.get("px")
    rt: RenderTarget | None = b.get("rt")
    return (px is not None and rt is not None
            and 0 <= px.pixel_x < rt.width
            and 0 <= px.pixel_y < rt.height
            and px.output_slot == rt.slot)

@guard("rs_rasterize_triangle_guard")
def _rs_rasterize_triangle_guard(b: Binding) -> bool:
    prim: Primitive | None = b.get("prim")
    return prim is not None and prim.topology == "triangle_list"

@guard("rs_rasterize_line_guard")
def _rs_rasterize_line_guard(b: Binding) -> bool:
    prim: Primitive | None = b.get("prim")
    return prim is not None and prim.topology == "line_list"

@guard("rs_rasterize_point_guard")
def _rs_rasterize_point_guard(b: Binding) -> bool:
    prim: Primitive | None = b.get("prim")
    return prim is not None and prim.topology not in ("triangle_list", "line_list")





def _add_all_places(net: PetriNet, cfg: dict) -> None:
    n_lanes = int(cfg.get("lanes", 4))
    mv = cfg.get("max_vertices", 65536)
    W  = int(cfg.get("render_width",  1920))
    H  = int(cfg.get("render_height", 1080))
    P  = net.add_place




    P("IA_vertex_buffer",      Vertex,            bound=mv)
    P("IA_index_buffer",       Index,             bound=mv)
    P("IA_input_layout",       ShaderBinding,     bound=1,
      initial=[_sb("IA", "InputLayout")])
    P("IA_topology",           ShaderBinding,     bound=1,
      initial=[_sb("IA", "triangle_list")])
    P("VS_input",              Vertex,            bound=mv)
    P("IA_assembled_primitive",Primitive,         bound=16384)
    P("IA_instance_data",      Vertex,            bound=4096)
    P("IA_strip_cut",          Index,             bound=1024)
    P("IA_draw_cmd",           ShaderBinding,     bound=1,
      initial=[_sb("IA", "Draw")])
    P("IA_indirect_args",      ConstantBuffer,    bound=1)
    P("IA_patch_size",         ShaderBinding,     bound=1,
      initial=[_sb("IA", "patch3")])
    P("IA_vertex_id",          Index,             bound=1,
      initial=[Index(value=0)])




    P("VS_cbuffer",            ConstantBuffer,    bound=14,
      initial=[ConstantBuffer(slot=0, size=16)])
    P("VS_shader",             ShaderBinding,     bound=1,
      initial=[_sb("VS")])
    P("VS_output",             Vertex,            bound=mv)
    for _l in range(n_lanes):
        P(f"VS_lane_{_l}_input",  Vertex,   bound=1)
        P(f"VS_lane_{_l}_output", Vertex,   bound=1)
        P(f"VS_lane_{_l}_slot",   LaneSlot, bound=1, initial=[LaneSlot(lane=_l)])
    P("HS_cbuffer",            ConstantBuffer,    bound=14,
      initial=[ConstantBuffer(slot=0, size=16)])
    P("HS_shader",             ShaderBinding,     bound=1,
      initial=[_sb("HS")])
    P("HS_input",              Patch,             bound=4096)
    P("HS_output",             Patch,             bound=4096)
    P("HS_tess_factor",        ConstantBuffer,    bound=4,
      initial=[ConstantBuffer(slot=0, size=1, data=(4.0,))])
    P("HS_bypass",             ShaderBinding,     bound=1,
      initial=[_sb("HS", "bypass")])
    P("HS_patch_constant",     Patch,             bound=4096)
    P("HS_sampler",            SamplerState,      bound=16,
      initial=[SamplerState(slot=0)])
    P("HS_fork_phase",         Patch,             bound=4096)
    P("HS_join_phase",         Patch,             bound=4096)
    P("HS_max_tess_factor",    ConstantBuffer,    bound=1,
      initial=[ConstantBuffer(slot=0, size=1, data=(64.0,))])
    P("HS_domain_type",        ShaderBinding,     bound=1,
      initial=[_sb("HS", "tri")])




    P("Tess_output",           DomainPoint,       bound=16384)
    P("Tess_factor_buffer",    ConstantBuffer,    bound=4,
      initial=[ConstantBuffer(slot=0, size=1, data=(4.0,))])
    P("Tess_topology_out",     ShaderBinding,     bound=1,
      initial=[_sb("Tess", "tri_cw")])
    P("Tess_partitioning",     ShaderBinding,     bound=1,
      initial=[_sb("Tess", "integer")])
    P("Tess_wind_order",       ShaderBinding,     bound=1,
      initial=[_sb("Tess", "cw")])
    P("Tess_bypass",           ShaderBinding,     bound=1,
      initial=[_sb("Tess", "bypass")])
    P("Tess_domain_pts_queue", DomainPoint,       bound=16384)




    P("DS_cbuffer",            ConstantBuffer,    bound=14,
      initial=[ConstantBuffer(slot=0, size=16)])
    P("DS_shader",             ShaderBinding,     bound=1,
      initial=[_sb("DS")])
    P("DS_output",             Vertex,            bound=mv)
    P("DS_sampler",            SamplerState,      bound=16,
      initial=[SamplerState(slot=0)])
    P("DS_patch_input",        Patch,             bound=4096)
    P("DS_bypass",             ShaderBinding,     bound=1,
      initial=[_sb("DS", "bypass")])
    P("DS_srv",                ConstantBuffer,    bound=128,
      initial=[ConstantBuffer(slot=0, size=1)])
    P("DS_uav",                ConstantBuffer,    bound=8)
    P("DS_clip_distance",      Vertex,            bound=4096)




    P("GS_input",              Primitive,         bound=8192)
    P("GS_cbuffer",            ConstantBuffer,    bound=14,
      initial=[ConstantBuffer(slot=0, size=16)])
    P("GS_shader",             ShaderBinding,     bound=1,
      initial=[_sb("GS")])
    P("GS_output",             Primitive,         bound=16384)
    P("GS_stream_select",      ShaderBinding,     bound=1,
      initial=[_sb("GS", "stream0")])
    P("GS_bypass",             ShaderBinding,     bound=1,
      initial=[_sb("GS", "bypass")])
    # D3D11_GS_MAX_OUTPUT_VERTEX_COUNT_ACROSS_INSTANCES = 1024
    P("GS_max_output_count",   ConstantBuffer,    bound=1,
      initial=[ConstantBuffer(slot=0, size=1, data=(1024.0,))])
    P("GS_sampler",            SamplerState,      bound=16,
      initial=[SamplerState(slot=0)])
    P("GS_instance_count",     ConstantBuffer,    bound=1,
      initial=[ConstantBuffer(slot=0, size=1, data=(1.0,))])
    P("GS_output_topology",    ShaderBinding,     bound=1,
      initial=[_sb("GS", "tri_strip")])
    P("GS_adjacency_input",    Primitive,         bound=4096)
    P("GS_cut_buffer",         ShaderBinding,     bound=1,
      initial=[_sb("GS", "no_cut")])




    P("SO_input",              Primitive,         bound=16384)
    P("SO_stream_0",           StreamRecord,      bound=4096)
    P("SO_stream_1",           StreamRecord,      bound=4096)
    P("SO_stream_2",           StreamRecord,      bound=4096)
    P("SO_stream_3",           StreamRecord,      bound=4096)
    P("SO_bypass",             ShaderBinding,     bound=1,
      initial=[_sb("SO", "bypass")])
    P("SO_write_offset_0",     Index,             bound=1,
      initial=[Index(value=0)])
    P("SO_write_offset_1",     Index,             bound=1,
      initial=[Index(value=0)])
    P("SO_write_offset_2",     Index,             bound=1,
      initial=[Index(value=0)])
    P("SO_write_offset_3",     Index,             bound=1,
      initial=[Index(value=0)])




    P("RS_input",              Primitive,         bound=16384)
    P("RS_viewport_0",         ViewportRect,      bound=1,
      initial=[ViewportRect(slot=0, x=0.0, y=0.0, width=W, height=H)])
    P("RS_viewport_1",         ViewportRect,      bound=1,
      initial=[ViewportRect(slot=1, x=0.0, y=H/2, width=W/2, height=H/2)])
    P("RS_scissor_0",          ScissorRect,       bound=1,
      initial=[ScissorRect(slot=0, left=0, top=0, right=W, bottom=H)])
    P("RS_scissor_1",          ScissorRect,       bound=1,
      initial=[ScissorRect(slot=1, left=0, top=H//2, right=W//2, bottom=H)])
    P("RS_rasterizer_state",   ShaderBinding,     bound=1,
      initial=[_sb("RS", "Default")])
    P("PS_input",              Fragment,          bound=mv)
    P("RS_clipped_primitive",  Primitive,         bound=8192)
    P("RS_culled_primitive",   Primitive,         bound=8192)
    P("RS_coverage_mask",      ShaderBinding,     bound=1,
      initial=[_sb("RS", "0xFFFF")])
    P("RS_fill_mode",          ShaderBinding,     bound=1,
      initial=[_sb("RS", "solid")])
    P("RS_cull_mode",          ShaderBinding,     bound=1,
      initial=[_sb("RS", "back")])
    P("RS_depth_clip_enable",  ShaderBinding,     bound=1,
      initial=[_sb("RS", "enabled")])
    P("RS_msaa_state",         ShaderBinding,     bound=1,
      initial=[_sb("RS", "1x")])
    P("RS_line_aa_state",      ShaderBinding,     bound=1,
      initial=[_sb("RS", "disabled")])
    P("RS_conservative_raster",ShaderBinding,     bound=1,
      initial=[_sb("RS", "disabled")])




    P("PS_cbuffer",            ConstantBuffer,    bound=14,
      initial=[ConstantBuffer(slot=0, size=16)])
    P("PS_shader",             ShaderBinding,     bound=1,
      initial=[_sb("PS")])
    P("PS_output",             Pixel,             bound=mv)
    P("PS_sampler",            SamplerState,      bound=16,
      initial=[SamplerState(slot=0)])
    P("PS_srv",                ConstantBuffer,    bound=128,
      initial=[ConstantBuffer(slot=0, size=1)])
    P("PS_uav",                ConstantBuffer,    bound=8)
    for _l in range(n_lanes):
        P(f"PS_lane_{_l}_input",  Fragment, bound=1)
        P(f"PS_lane_{_l}_output", Pixel,    bound=1)
        P(f"PS_lane_{_l}_slot",   LaneSlot, bound=1, initial=[LaneSlot(lane=_l)])
    P("PS_discard_signal",     ShaderBinding,     bound=1)
    # Sanctioned-discard sinks.  Without them PS_discard and the OM
    # depth/stencil-fail discards consume their token and produce nothing,
    # so a legitimately discarded fragment is indistinguishable from a lost
    # one and token conservation (L1) cannot account for it.
    P("PS_discarded_fragment", Fragment,          bound=None)
    P("OM_blend_state",        BlendState,        bound=1,
      initial=[BlendState(blend_enable=False)])
    P("OM_depth_stencil_state",DepthStencilState, bound=1,
      initial=[DepthStencilState(depth_enable=True, depth_write_mask=True,
                                  depth_func="LESS")])
    for i in range(8):
        P(f"OM_rt_slot_{i}",   RenderTarget,      bound=1,
          initial=[RenderTarget(slot=i, width=W, height=H)])
    P("OM_dsv",                RenderTarget,      bound=1,
      initial=[RenderTarget(slot=-1, width=W, height=H)])
    for i in range(8):
        P(f"OM_written_pixel_{i}", Pixel,         bound=None)
    P("OM_depth_buffer",       Pixel,             bound=None)
    P("OM_stencil_buffer",     Pixel,             bound=None)
    P("OM_rt_active",          ShaderBinding,     bound=1,
      initial=[_sb("OM", "rt_active")])
    P("OM_rt_in_use",          ShaderBinding,     bound=1)
    P("OM_discarded_pixel",    Pixel,             bound=None)





def _add_ia_transitions(net: PetriNet, _cfg: dict) -> None:
    T  = net.add_transition
    IA = net.add_input_arc
    OA = net.add_output_arc


    T("IA_fetch_vertex")
    _ro(net, "IA_input_layout", "IA_fetch_vertex", "layout")
    _ro(net, "IA_topology",     "IA_fetch_vertex", "topo")
    IA("IA_vertex_buffer", "IA_fetch_vertex", _p("v"))
    OA("VS_input",         "IA_fetch_vertex", _p("v"))


    T("IA_fetch_indexed")
    _ro(net, "IA_input_layout", "IA_fetch_indexed", "layout")
    IA("IA_index_buffer",  "IA_fetch_indexed", _p("idx"))
    IA("IA_vertex_buffer", "IA_fetch_indexed", _p("v"))
    OA("VS_input",         "IA_fetch_indexed", _p("v"))


    T("IA_assemble_triangle")
    _ro(net, "IA_topology", "IA_assemble_triangle", "topo")
    for var in ("v0", "v1", "v2"):
        IA("VS_input", "IA_assemble_triangle", _p(var))
    OA("IA_assembled_primitive", "IA_assemble_triangle",
       lambda b: Multiset.of(Primitive(vertices=(b["v0"], b["v1"], b["v2"]),
                                       topology="triangle_list")))

    T("IA_assemble_line")
    for var in ("v0", "v1"):
        IA("VS_input", "IA_assemble_line", _p(var))
    OA("IA_assembled_primitive", "IA_assemble_line",
       lambda b: Multiset.of(Primitive(vertices=(b["v0"], b["v1"]),
                                       topology="line_list")))

    T("IA_assemble_point")
    IA("VS_input",               "IA_assemble_point", _p("v"))
    OA("IA_assembled_primitive", "IA_assemble_point",
       lambda b: Multiset.of(Primitive(vertices=(b["v"],),
                                       topology="point_list")))


    T("IA_apply_strip_cut")
    _ro(net, "IA_vertex_buffer", "IA_apply_strip_cut", "vb")
    IA("IA_strip_cut",  "IA_apply_strip_cut", _p("idx"))
    OA("IA_vertex_id",  "IA_apply_strip_cut", _p("idx"))


    T("IA_indirect_draw")
    _ro(net, "IA_indirect_args",   "IA_indirect_draw", "args")
    _ro(net, "IA_vertex_buffer",   "IA_indirect_draw", "vb")
    _ro(net, "IA_input_layout",    "IA_indirect_draw", "layout")


    T("IA_advance_instance")
    IA("IA_instance_data", "IA_advance_instance", _p("v"))
    OA("VS_input",         "IA_advance_instance", _p("v"))


    T("IA_emit_draw_cmd")
    _ro(net, "IA_draw_cmd", "IA_emit_draw_cmd", "cmd")
    _ro(net, "IA_topology", "IA_emit_draw_cmd", "topo")
    OA("VS_input", "IA_emit_draw_cmd",
       lambda _b: Multiset.of(Vertex(vertex_id=0, clip_pos=(1.0, 0.0, 0.0, 1.0))))





def _add_vs_transitions(net: PetriNet, _cfg: dict) -> None:
    T  = net.add_transition
    IA = net.add_input_arc
    OA = net.add_output_arc

    for lane in range(int(_cfg.get("lanes", 4))):
        slot_expr = _p("slot")


        T(f"VS_dispatch_lane_{lane}")
        IA("VS_input",               f"VS_dispatch_lane_{lane}", _p("v"))
        IA(f"VS_lane_{lane}_slot",   f"VS_dispatch_lane_{lane}", slot_expr)
        OA(f"VS_lane_{lane}_input",  f"VS_dispatch_lane_{lane}", _p("v"))


        T(f"VS_exec_lane_{lane}", guard=_vs_exec_guard)
        IA(f"VS_lane_{lane}_input",  f"VS_exec_lane_{lane}", _p("v"))
        _ro(net, "VS_cbuffer", f"VS_exec_lane_{lane}", "cbuf")
        _ro(net, "VS_shader",  f"VS_exec_lane_{lane}", "vs_shader")
        OA(f"VS_lane_{lane}_output", f"VS_exec_lane_{lane}",
           lambda b: Multiset.of(_vs_transform(b["v"], b["cbuf"])))


        T(f"VS_collect_lane_{lane}")
        IA(f"VS_lane_{lane}_output", f"VS_collect_lane_{lane}", _p("v"))
        OA("VS_output",              f"VS_collect_lane_{lane}", _p("v"))
        OA(f"VS_lane_{lane}_slot",   f"VS_collect_lane_{lane}",
           lambda _b, _l=lane: Multiset.of(LaneSlot(lane=_l)))





def _add_hs_transitions(net: PetriNet, _cfg: dict) -> None:
    T  = net.add_transition
    IA = net.add_input_arc
    OA = net.add_output_arc


    T("HS_batch_patch")
    for var in ("v0", "v1", "v2"):
        IA("VS_output", "HS_batch_patch", _p(var))
    _ro(net, "HS_patch_size" if "HS_patch_size" in net.places else "IA_patch_size",
        "HS_batch_patch", "psize")
    OA("HS_input", "HS_batch_patch",
       lambda b: Multiset.of(Patch(vertices=(b["v0"], b["v1"], b["v2"]))))


    T("HS_exec_hull", guard=_hs_exec_guard)
    IA("HS_input",     "HS_exec_hull", _p("patch"))
    _ro(net, "HS_cbuffer",     "HS_exec_hull", "cbuf")
    _ro(net, "HS_shader",      "HS_exec_hull", "hs_shader")
    _ro(net, "HS_domain_type", "HS_exec_hull", "dtype")
    OA("HS_fork_phase", "HS_exec_hull", _p("patch"))


    T("HS_exec_fork")
    IA("HS_fork_phase", "HS_exec_fork", _p("patch"))
    _ro(net, "HS_cbuffer", "HS_exec_fork", "cbuf")
    _ro(net, "HS_shader",  "HS_exec_fork", "hs_shader")
    OA("HS_join_phase", "HS_exec_fork", _p("patch"))


    T("HS_exec_join")
    IA("HS_join_phase",  "HS_exec_join", _p("patch"))
    _ro(net, "HS_cbuffer",         "HS_exec_join", "cbuf")
    _ro(net, "HS_shader",          "HS_exec_join", "hs_shader")
    _ro(net, "HS_max_tess_factor", "HS_exec_join", "maxtf")
    OA("HS_output",      "HS_exec_join", _p("patch"))


    T("HS_emit_tess_factors", guard=_tess_factor_valid)
    _ro(net, "HS_tess_factor", "HS_emit_tess_factors", "tf")


    T("HS_emit_patch_constants")
    IA("HS_output",        "HS_emit_patch_constants", _p("patch"))
    OA("HS_patch_constant","HS_emit_patch_constants", _p("patch"))
    OA("HS_output",        "HS_emit_patch_constants", _p("patch"))


    T("HS_bypass_exec")
    IA("VS_output",   "HS_bypass_exec", _p("v"))
    _ro(net, "HS_bypass", "HS_bypass_exec", "bypass")
    OA("GS_input",    "HS_bypass_exec",
       lambda b: Multiset.of(Primitive(vertices=(b["v"],), topology="point_list")))


    T("HS_cull_patch")
    IA("HS_output",      "HS_cull_patch", _p("patch"))
    IA("HS_tess_factor", "HS_cull_patch", _p("tf"))



    T("HS_validate_domain")
    _ro(net, "HS_domain_type", "HS_validate_domain", "dtype")


    T("HS_clamp_tess_factor")
    IA("HS_tess_factor",        "HS_clamp_tess_factor", _p("tf"))
    _ro(net, "HS_max_tess_factor","HS_clamp_tess_factor", "maxbnd")
    OA("HS_tess_factor",        "HS_clamp_tess_factor",
       lambda b: Multiset.of(ConstantBuffer(
           slot=b["tf"].slot, size=b["tf"].size,
           data=tuple(min(f, b["maxbnd"].data[0]) for f in b["tf"].data))))


    T("HS_output_control_pts")
    IA("HS_patch_constant", "HS_output_control_pts", _p("patch"))
    OA("HS_output",         "HS_output_control_pts", _p("patch"))


    T("HS_release_resources")
    _ro(net, "HS_output",  "HS_release_resources", "patch")
    _ro(net, "HS_sampler", "HS_release_resources", "samp")





def _add_tess_transitions(net: PetriNet, _cfg: dict) -> None:
    T  = net.add_transition
    IA = net.add_input_arc
    OA = net.add_output_arc

    def _tess_exec(name: str) -> None:
        T(name, guard=_tess_factor_valid)
        IA("HS_output",             name, _p("patch"))
        _ro(net, "Tess_factor_buffer",  name, "tf")
        _ro(net, "Tess_partitioning",   name, "part")
        _ro(net, "Tess_wind_order",     name, "wind")
        _ro(net, "Tess_topology_out",   name, "topo")
        OA("Tess_domain_pts_queue", name,
           lambda b, _p2=name: Multiset.of(
               DomainPoint(patch=b["patch"], u=0.5, v=0.25, w=0.25)))

    _tess_exec("Tess_exec_tri")
    _tess_exec("Tess_exec_quad")
    _tess_exec("Tess_exec_isoline")


    for part_name in ("Tess_fractional_even", "Tess_fractional_odd", "Tess_integer"):
        T(part_name)
        _ro(net, "Tess_partitioning", part_name, "part")


    T("Tess_bypass_exec")
    IA("HS_output",   "Tess_bypass_exec", _p("patch"))
    _ro(net, "Tess_bypass", "Tess_bypass_exec", "bypass")
    OA("Tess_output", "Tess_bypass_exec",
       lambda b: Multiset.of(DomainPoint(patch=b["patch"], u=0.5, v=0.5)))


    T("Tess_emit_domain_point")
    IA("Tess_domain_pts_queue", "Tess_emit_domain_point", _p("dp"))
    OA("Tess_output",           "Tess_emit_domain_point", _p("dp"))





def _add_ds_transitions(net: PetriNet, _cfg: dict) -> None:
    T  = net.add_transition
    IA = net.add_input_arc
    OA = net.add_output_arc


    T("DS_exec", guard=_ds_exec_guard)
    IA("Tess_output",   "DS_exec", _p("dp"))
    IA("DS_patch_input","DS_exec", _p("patch"))
    _ro(net, "DS_cbuffer", "DS_exec", "cbuf")
    _ro(net, "DS_shader",  "DS_exec", "ds_shader")
    OA("DS_patch_input","DS_exec", _p("patch"))
    OA("DS_output",     "DS_exec", _p("v_out"))


    T("DS_read_sampler")
    _ro(net, "DS_output",  "DS_read_sampler", "v")
    _ro(net, "DS_sampler", "DS_read_sampler", "samp")
    _ro(net, "DS_srv",     "DS_read_sampler", "srv")

    T("DS_read_srv")
    _ro(net, "DS_output",  "DS_read_srv", "v")
    _ro(net, "DS_srv",     "DS_read_srv", "srv")

    T("DS_uav_write")
    _ro(net, "DS_output",  "DS_uav_write", "v")
    _ro(net, "DS_uav",     "DS_uav_write", "uav")


    T("DS_emit_clip_distance")
    IA("DS_output",         "DS_emit_clip_distance", _p("v"))
    OA("DS_clip_distance",  "DS_emit_clip_distance", _p("v"))
    OA("DS_output",         "DS_emit_clip_distance", _p("v"))


    T("DS_bypass_exec")
    IA("Tess_output",  "DS_bypass_exec", _p("dp"))
    _ro(net, "DS_bypass", "DS_bypass_exec", "bypass")
    OA("DS_output",    "DS_bypass_exec",
       lambda b: Multiset.of(Vertex(
           vertex_id=b["dp"].domain_id,
           clip_pos=(b["dp"].u * 2 - 1, b["dp"].v * 2 - 1, 0.5, 1.0))))


    T("DS_validate_output")
    IA("DS_output",  "DS_validate_output", _p("v"))
    OA("DS_output",  "DS_validate_output", _p("v"))


    T("DS_assemble_primitive")
    for var in ("v0", "v1", "v2"):
        IA("DS_output", "DS_assemble_primitive", _p(var))
    OA("GS_input", "DS_assemble_primitive",
       lambda b: Multiset.of(Primitive(vertices=(b["v0"], b["v1"], b["v2"]),
                                       topology="triangle_list")))


    T("DS_read_patch")
    _ro(net, "DS_patch_input", "DS_read_patch", "patch")





def _add_gs_transitions(net: PetriNet, _cfg: dict) -> None:
    T  = net.add_transition
    IA = net.add_input_arc
    OA = net.add_output_arc


    T("GS_exec")
    IA("GS_input",   "GS_exec", _p("prim"))
    _ro(net, "GS_cbuffer",          "GS_exec", "cbuf")
    _ro(net, "GS_shader",           "GS_exec", "gs_shader")
    _ro(net, "GS_max_output_count", "GS_exec", "maxout")
    OA("GS_output",  "GS_exec", _p("prim"))


    T("GS_emit_vertex")
    IA("GS_output",  "GS_emit_vertex", _p("prim"))
    OA("GS_output",  "GS_emit_vertex", _p("prim"))


    T("GS_cut_strip")
    _ro(net, "GS_output",     "GS_cut_strip", "prim")
    _ro(net, "GS_cut_buffer", "GS_cut_strip", "cut")


    T("GS_emit_to_stream_0")
    IA("GS_output",  "GS_emit_to_stream_0", _p("prim"))
    _ro(net, "GS_stream_select", "GS_emit_to_stream_0", "sel")
    OA("SO_input",   "GS_emit_to_stream_0", _p("prim"))
    OA("RS_input",   "GS_emit_to_stream_0", _p("prim"))

    for s in (1, 2, 3):
        T(f"GS_emit_to_stream_{s}")
        IA("GS_output", f"GS_emit_to_stream_{s}", _p("prim"))
        OA("SO_input",  f"GS_emit_to_stream_{s}", _p("prim"))


    T("GS_bypass_exec")
    IA("GS_input",   "GS_bypass_exec", _p("prim"))
    _ro(net, "GS_bypass", "GS_bypass_exec", "bypass")
    OA("RS_input",   "GS_bypass_exec", _p("prim"))


    T("GS_assemble_adjacency")
    IA("GS_adjacency_input", "GS_assemble_adjacency", _p("prim"))
    OA("GS_input",           "GS_assemble_adjacency", _p("prim"))


    T("GS_instance_step")
    _ro(net, "GS_instance_count", "GS_instance_step", "inst")
    IA("GS_input",   "GS_instance_step", _p("prim"))
    OA("GS_input",   "GS_instance_step", _p("prim"))


    T("GS_validate_output_count")
    _ro(net, "GS_max_output_count", "GS_validate_output_count", "maxout")
    _ro(net, "GS_output_topology",  "GS_validate_output_count", "otopo")


    T("GS_release_resources")
    _ro(net, "GS_sampler", "GS_release_resources", "samp")
    _ro(net, "GS_cbuffer", "GS_release_resources", "cbuf")





def _add_so_transitions(net: PetriNet, _cfg: dict) -> None:
    T  = net.add_transition
    IA = net.add_input_arc
    OA = net.add_output_arc

    for s in range(4):

        T(f"SO_write_stream_{s}")
        IA("SO_input",             f"SO_write_stream_{s}", _p("prim"))
        IA(f"SO_write_offset_{s}", f"SO_write_stream_{s}", _p("off"))
        OA(f"SO_stream_{s}",       f"SO_write_stream_{s}",
           lambda b, _s=s: Multiset.of(
               StreamRecord(stream_index=_s,
                            primitive=b["prim"],
                            byte_offset=b["off"].value)))
        OA(f"SO_write_offset_{s}", f"SO_write_stream_{s}",
           lambda b: Multiset.of(Index(value=b["off"].value + 1)))



    for s in range(3):
        T(f"SO_advance_offset_{s}")
        IA(f"SO_write_offset_{s}", f"SO_advance_offset_{s}", _p("off"))
        OA(f"SO_write_offset_{s}", f"SO_advance_offset_{s}",
           lambda b: Multiset.of(Index(value=b["off"].value)))


    T("SO_bypass_exec")
    IA("SO_input",   "SO_bypass_exec", _p("prim"))
    _ro(net, "SO_bypass", "SO_bypass_exec", "bypass")
    OA("RS_input",   "SO_bypass_exec", _p("prim"))





def _add_rs_transitions(net: PetriNet, _cfg: dict) -> None:
    T  = net.add_transition
    IA = net.add_input_arc
    OA = net.add_output_arc

    def _clip_plane(name: str) -> None:
        T(name)
        _ro(net, "RS_depth_clip_enable", name, "dce")
        IA("RS_input",            name, _p("prim"))
        OA("RS_clipped_primitive",name, _p("prim"))

    def _clip_discard(name: str) -> None:
        T(name)
        IA("RS_input",            name, _p("prim"))
        OA("RS_culled_primitive", name, _p("prim"))


    _clip_plane("RS_clip_near");   _clip_plane("RS_clip_far")
    _clip_plane("RS_clip_left");   _clip_plane("RS_clip_right")
    _clip_plane("RS_clip_top");    _clip_plane("RS_clip_bottom")


    for cull in ("RS_cull_back", "RS_cull_front"):
        T(cull)
        IA("RS_clipped_primitive", cull, _p("prim"))
        _ro(net, "RS_cull_mode",   cull, "cmode")
        OA("RS_culled_primitive",  cull, _p("prim"))


    for vp_slot in (0, 1):
        name = f"RS_viewport_transform_{vp_slot}"
        T(name)
        IA("RS_clipped_primitive", name, _p("prim"))
        _ro(net, f"RS_viewport_{vp_slot}", name, "vp")
        OA("RS_clipped_primitive", name, _p("prim"))


    for sc_slot in (0, 1):
        name = f"RS_scissor_test_{sc_slot}"
        T(name)
        IA("RS_clipped_primitive", name, _p("prim"))
        _ro(net, f"RS_scissor_{sc_slot}", name, "sc")
        OA("RS_clipped_primitive", name, _p("prim"))


    T("RS_check_msaa")
    IA("RS_clipped_primitive", "RS_check_msaa", _p("prim"))
    _ro(net, "RS_msaa_state",         "RS_check_msaa", "msaa")
    OA("RS_clipped_primitive", "RS_check_msaa", _p("prim"))

    T("RS_check_conservative_raster")
    IA("RS_clipped_primitive",          "RS_check_conservative_raster", _p("prim"))
    _ro(net, "RS_conservative_raster",  "RS_check_conservative_raster", "cr")
    OA("RS_clipped_primitive",          "RS_check_conservative_raster", _p("prim"))


    T("RS_rasterize_triangle", guard=_rs_rasterize_triangle_guard)
    IA("RS_clipped_primitive",  "RS_rasterize_triangle", _p("prim"))
    _ro(net, "RS_rasterizer_state", "RS_rasterize_triangle", "rs")
    _ro(net, "RS_coverage_mask",    "RS_rasterize_triangle", "cov")
    _ro(net, "RS_fill_mode",        "RS_rasterize_triangle", "fill")
    OA("PS_input", "RS_rasterize_triangle",
       lambda b: Multiset.of(
           Fragment(pixel_x=int(b["prim"].vertices[0].clip_pos[0]),
                    pixel_y=int(b["prim"].vertices[0].clip_pos[1]),
                    depth=b["prim"].vertices[0].clip_pos[2])))

    T("RS_rasterize_line", guard=_rs_rasterize_line_guard)
    IA("RS_clipped_primitive", "RS_rasterize_line", _p("prim"))
    _ro(net, "RS_line_aa_state", "RS_rasterize_line", "laa")
    OA("PS_input", "RS_rasterize_line",
       lambda b: Multiset.of(
           Fragment(pixel_x=0, pixel_y=0, depth=0.5)))

    T("RS_rasterize_point", guard=_rs_rasterize_point_guard)
    IA("RS_clipped_primitive", "RS_rasterize_point", _p("prim"))
    OA("PS_input", "RS_rasterize_point",
       lambda b: Multiset.of(
           Fragment(pixel_x=0, pixel_y=0, depth=0.5)))


    T("RS_interpolate_attributes")
    IA("PS_input",  "RS_interpolate_attributes", _p("frag"))
    OA("PS_input",  "RS_interpolate_attributes", _p("frag"))





def _add_ps_transitions(net: PetriNet, _cfg: dict) -> None:
    T  = net.add_transition
    IA = net.add_input_arc
    OA = net.add_output_arc

    slot_expr = _p("slot")
    for lane in range(int(_cfg.get("lanes", 4))):

        T(f"PS_dispatch_lane_{lane}")
        IA("PS_input",               f"PS_dispatch_lane_{lane}", _p("frag"))
        IA(f"PS_lane_{lane}_slot",   f"PS_dispatch_lane_{lane}", slot_expr)
        OA(f"PS_lane_{lane}_input",  f"PS_dispatch_lane_{lane}", _p("frag"))


        T(f"PS_exec_lane_{lane}", guard=_ps_exec_guard)
        IA(f"PS_lane_{lane}_input",  f"PS_exec_lane_{lane}", _p("frag"))
        _ro(net, "PS_cbuffer", f"PS_exec_lane_{lane}", "cbuf")
        _ro(net, "PS_shader",  f"PS_exec_lane_{lane}", "ps_shader")
        OA(f"PS_lane_{lane}_output", f"PS_exec_lane_{lane}",
           lambda b: Multiset.of(_ps_transform(b["frag"])))


        T(f"PS_collect_lane_{lane}")
        IA(f"PS_lane_{lane}_output", f"PS_collect_lane_{lane}", _p("px"))
        OA("PS_output",              f"PS_collect_lane_{lane}", _p("px"))
        OA(f"PS_lane_{lane}_slot",   f"PS_collect_lane_{lane}",
           lambda _b, _l=lane: Multiset.of(LaneSlot(lane=_l)))


    T("PS_sample_texture_0")
    _ro(net, "PS_input",   "PS_sample_texture_0", "frag")
    _ro(net, "PS_sampler", "PS_sample_texture_0", "samp")
    _ro(net, "PS_srv",     "PS_sample_texture_0", "srv")

    T("PS_sample_texture_1")
    _ro(net, "PS_input",   "PS_sample_texture_1", "frag")
    _ro(net, "PS_sampler", "PS_sample_texture_1", "samp")
    _ro(net, "PS_srv",     "PS_sample_texture_1", "srv")


    T("PS_uav_write")
    _ro(net, "PS_input", "PS_uav_write", "frag")
    _ro(net, "PS_uav",   "PS_uav_write", "uav")


    T("PS_discard")
    IA("PS_input", "PS_discard", _p("frag"))
    _ro(net, "PS_discard_signal", "PS_discard", "sig")
    OA("PS_discarded_fragment", "PS_discard", _p("frag"))






def _add_om_transitions(net: PetriNet, _cfg: dict) -> None:
    T  = net.add_transition
    IA = net.add_input_arc
    OA = net.add_output_arc


    T("OM_depth_test", guard=_om_depth_less)
    IA("PS_output",              "OM_depth_test", _p("px"))
    _ro(net, "OM_depth_stencil_state", "OM_depth_test", "dss")
    _ro(net, "OM_dsv",                 "OM_depth_test", "rt")
    OA("PS_output",              "OM_depth_test", _p("px"))


    T("OM_stencil_test")
    IA("PS_output",              "OM_stencil_test", _p("px"))
    _ro(net, "OM_depth_stencil_state", "OM_stencil_test", "dss")
    OA("PS_output",              "OM_stencil_test", _p("px"))


    T("OM_depth_write")
    IA("PS_output",       "OM_depth_write", _p("px"))
    _ro(net, "OM_depth_stencil_state", "OM_depth_write", "dss")
    _ro(net, "OM_dsv",    "OM_depth_write", "rt")
    OA("PS_output",       "OM_depth_write", _p("px"))
    OA("OM_depth_buffer", "OM_depth_write", _p("px"))


    T("OM_stencil_write")
    IA("PS_output",         "OM_stencil_write", _p("px"))
    _ro(net, "OM_depth_stencil_state", "OM_stencil_write", "dss")
    OA("OM_stencil_buffer", "OM_stencil_write", _p("px"))
    OA("PS_output",         "OM_stencil_write", _p("px"))


    for i in range(8):
        T(f"OM_blend_rt_{i}", guard=_om_pixel_in_rt)
        IA("PS_output",       f"OM_blend_rt_{i}", _p("px"))
        _ro(net, "OM_blend_state",   f"OM_blend_rt_{i}", "bs")
        _ro(net, f"OM_rt_slot_{i}",  f"OM_blend_rt_{i}", "rt")
        OA("PS_output",       f"OM_blend_rt_{i}", _p("px"))


    for i in range(8):
        T(f"OM_write_rt_{i}", guard=_om_pixel_in_rt)
        IA("PS_output",            f"OM_write_rt_{i}", _p("px"))
        IA("OM_rt_active",         f"OM_write_rt_{i}", _p("ra"))
        _ro(net, f"OM_rt_slot_{i}",f"OM_write_rt_{i}", "rt")
        _ro(net, "OM_blend_state", f"OM_write_rt_{i}", "bs")
        OA(f"OM_written_pixel_{i}",f"OM_write_rt_{i}", _p("px"))
        OA("OM_rt_in_use",         f"OM_write_rt_{i}", _p("ra"))

    T("OM_release_rt")
    IA("OM_rt_in_use", "OM_release_rt", _p("ra"))
    OA("OM_rt_active", "OM_release_rt", _p("ra"))


    T("OM_discard_depth_fail")
    IA("PS_output", "OM_discard_depth_fail", _p("px"))
    _ro(net, "OM_depth_stencil_state", "OM_discard_depth_fail", "dss")
    OA("OM_discarded_pixel", "OM_discard_depth_fail", _p("px"))



    T("OM_discard_stencil_fail")
    IA("PS_output", "OM_discard_stencil_fail", _p("px"))
    _ro(net, "OM_depth_stencil_state", "OM_discard_stencil_fail", "dss")
    OA("OM_discarded_pixel", "OM_discard_stencil_fail", _p("px"))






def build_full_pipeline(config: dict[str, Any] | None = None) -> PetriNet:
    cfg = config or {}
    net = PetriNet(name="DX11_Full")

    _add_all_places(net, cfg)

    _add_ia_transitions(net, cfg)
    _add_vs_transitions(net, cfg)
    _add_hs_transitions(net, cfg)
    _add_tess_transitions(net, cfg)
    _add_ds_transitions(net, cfg)
    _add_gs_transitions(net, cfg)
    _add_so_transitions(net, cfg)
    _add_rs_transitions(net, cfg)
    _add_ps_transitions(net, cfg)
    _add_om_transitions(net, cfg)

    p = len(net.places)
    t = len(net.transitions)
    assert p > 100, f"Expected > 100 places, got {p}"
    assert t > 100, f"Expected > 100 transitions, got {t}"
    return net





if __name__ == "__main__":
    net = build_full_pipeline()

    print(f"\nDX11 Full Pipeline CPN")
    print(f"{'Places':<16}: {len(net.places)}")
    print(f"{'Transitions':<16}: {len(net.transitions)}")
    print(f"{'Arcs':<16}: {len(net.arcs)}")

    print("\nPer-stage breakdown:")
    stages = ["IA", "VS", "HS", "Tess", "DS", "GS", "SO", "RS", "PS", "OM"]
    for stage in stages:
        p_count = sum(1 for n in net.places      if n.startswith(stage + "_") or n.startswith(stage.lower() + "_"))
        t_count = sum(1 for n in net.transitions if n.startswith(stage + "_") or n.startswith(stage.lower() + "_"))
        print(f"  {stage:<8}  places={p_count:<4}  transitions={t_count}")

    print("\nParallel structures:")
    for prefix in ("VS_lane_", "PS_lane_", "OM_rt_slot_", "SO_stream_",
                   "RS_viewport_", "RS_scissor_"):
        count = sum(1 for n in net.places if n.startswith(prefix))
        if count:
            print(f"  {prefix}*  → {count} parallel places")
    for prefix in ("VS_exec_lane_", "VS_dispatch_lane_", "VS_collect_lane_",
                   "PS_exec_lane_",  "PS_dispatch_lane_",  "PS_collect_lane_",
                   "OM_blend_rt_", "OM_write_rt_",
                   "SO_write_stream_", "GS_emit_to_stream_"):
        count = sum(1 for n in net.transitions if n.startswith(prefix))
        if count:
            print(f"  {prefix}*  → {count} parallel transitions")

    print("\nInitial tokens (non-empty places at t=0):")
    for name, place in net.places.items():
        if place.token_count() > 0:
            print(f"  {name:<36} [{place.token_count()} tok]")
