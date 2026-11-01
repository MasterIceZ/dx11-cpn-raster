
from __future__ import annotations

from typing import Any

from pipeline.petri.colored import (
    Binding,
    Fragment,
    Pixel,
    guard,
)
from pipeline.petri.net import Multiset, PetriNet
from pipeline.stages.vertex_shader import ConstantBuffer, ShaderBinding
from models.dx11_full_pipeline import LaneSlot, SamplerState

N_LANES = 4

# MEM_free is a 1-bounded semaphore that always carries this single fixed
# token — there is exactly one shared port, so "free" and "this token is
# present" coincide. Acquire/release arcs reference it by value (not by an
# input-bound variable) since no transition ever observes *which* token is
# in MEM_free, only whether one is present.
_MEM_PORT_TOKEN = ShaderBinding(stage="PS", name="uav_port", valid=True)
_free_expr = lambda _b: Multiset.of(_MEM_PORT_TOKEN)


def _p(var: str):
    return lambda b, _v=var: Multiset.of(b[_v])


def _ro(net: PetriNet, place: str, transition: str, var: str) -> None:
    net.add_input_arc(place, transition, _p(var))
    net.add_output_arc(place, transition, _p(var))


def ps_transform(frag: Fragment) -> Pixel:
    return Pixel(
        pixel_x=frag.pixel_x,
        pixel_y=frag.pixel_y,
        color=(1.0, 0.0, 0.0, 1.0),
        depth=frag.depth,
        output_slot=0,
    )


@guard("ps_exec_guard")
def ps_exec_guard(b: Binding) -> bool:
    s = b.get("ps_shader"); c = b.get("cbuf"); frag = b.get("frag")
    if not (s and c and frag):
        return False
    if not (s.valid and 0 <= c.slot <= 13 and 0.0 <= frag.depth <= 1.0):
        return False
    px = ps_transform(frag)
    return 0 <= px.output_slot <= 7


def build_subnet(config: dict[str, Any] | None = None) -> PetriNet:
    """Standalone Pixel Shader (PS) subnet — *with* shared-resource contention.

    ``PS_input``/``PS_output`` are wired in as PS's input/output interfaces,
    exactly as in the composed pipeline, and the four 1-bounded
    dispatch -> exec -> collect lanes mirror ``models.dx11_full_pipeline``'s
    ``_add_ps_transitions``/``vertex_shader``-style conventions.

    Unlike OM (whose mutual-exclusion mechanism already exists via
    ``OM_rt_active``/``OM_rt_in_use``) and RS (whose queue is simply
    ``PS_input``), the composed model has *no* mechanism modelling contention
    over a shared PS resource — ``PS_uav`` is only ever touched through a
    read-only probe arc (``PS_uav_write``/``_ro``). The paper's L5 ("PS
    Starvation Freedom") atomic propositions (``_ps_waiting``/``_mem_free``/
    ``_ps_has_mem`` in ``pipeline.verifier.dx11_ltl``) referenced places
    (``PS_mem_wait``, ``MEM_free``, ``PS_mem_acquired``) that consequently
    never existed anywhere in the model, making every reported L5 result —
    not just the VS-subnet table the professor flagged — vacuously true.

    This subnet adds the smallest real mechanism that gives those places
    actual contention semantics: a classic request -> acquire -> release
    mutual-exclusion protocol over a single shared port (modelling, e.g., the
    UAV memory port a real PS wave's lanes serialise atomic read-modify-write
    accesses through). Each lane

      1. *requests* the port — gated on the lane actually holding fragment
         work (``PS_lane_N_input``), so contention only arises from genuine
         PS activity — consuming its own ``PS_mem_req_slot_N`` ticket and
         depositing a ``LaneSlot(lane=N)`` token in ``PS_mem_wait``;
      2. *acquires* the port — consuming its waiting ticket together with the
         ``MEM_free`` token, depositing it in ``PS_mem_acquired``
         (1-bounded: only one lane may hold the port at a time, the
         "PS_has_mem" mutual-exclusion invariant L5 depends on); then
      3. *releases* the port — returning the ``MEM_free`` token and its own
         ``PS_mem_req_slot_N`` ticket so the lane may contend again later.

    With this protocol in place, "PS waiting" and "memory free" are reachable
    simultaneously, "PS has memory" is reachable from every such state (the
    acquire transition is always enabled there), and the mutual-exclusion
    bound (``PS_mem_acquired`` is 1-bounded) is structural — so L5 becomes a
    genuine, checkable starvation-freedom property of this subnet rather than
    a vacuous pass over nonexistent places.
    """
    cfg: dict[str, Any] = config or {}
    max_fragments: int = cfg.get("max_fragments", 65_536)
    max_pixels: int = cfg.get("max_pixels", 65_536)
    shader_name: str = cfg.get("shader_name", "DefaultPS")

    net = PetriNet(name="PS")

    net.add_place("PS_input", Fragment, bound=max_fragments)
    net.add_place("PS_output", Pixel, bound=max_pixels)
    net.add_place("PS_cbuffer", ConstantBuffer, bound=14,
                  initial=[ConstantBuffer(slot=0, size=16)])
    net.add_place("PS_shader", ShaderBinding, bound=1,
                  initial=[ShaderBinding(stage="PS", name=shader_name, valid=True)])
    net.add_place("PS_sampler", SamplerState, bound=16,
                  initial=[SamplerState(slot=0)])
    net.add_place("PS_srv", ConstantBuffer, bound=128,
                  initial=[ConstantBuffer(slot=0, size=1)])
    net.add_place("PS_uav", ConstantBuffer, bound=8)
    net.add_place("PS_discard_signal", ShaderBinding, bound=1)

    for lane in range(N_LANES):
        net.add_place(f"PS_lane_{lane}_input", Fragment, bound=1)
        net.add_place(f"PS_lane_{lane}_output", Pixel, bound=1)
        net.add_place(f"PS_lane_{lane}_slot", LaneSlot, bound=1,
                      initial=[LaneSlot(lane=lane)])

    # ------------------------------------------------------------------
    # New: minimal PS shared-resource contention machinery for L5
    # ------------------------------------------------------------------
    for lane in range(N_LANES):
        net.add_place(f"PS_mem_req_slot_{lane}", LaneSlot, bound=1,
                      initial=[LaneSlot(lane=lane)])
    net.add_place("PS_mem_wait", LaneSlot, bound=N_LANES)
    net.add_place("MEM_free", ShaderBinding, bound=1, initial=[_MEM_PORT_TOKEN])
    net.add_place("PS_mem_acquired", LaneSlot, bound=1)

    slot_expr = _p("slot")
    for lane in range(N_LANES):
        net.add_transition(f"PS_dispatch_lane_{lane}")
        net.add_input_arc("PS_input", f"PS_dispatch_lane_{lane}", _p("frag"))
        net.add_input_arc(f"PS_lane_{lane}_slot", f"PS_dispatch_lane_{lane}", slot_expr)
        net.add_output_arc(f"PS_lane_{lane}_input", f"PS_dispatch_lane_{lane}", _p("frag"))

        net.add_transition(f"PS_exec_lane_{lane}", guard=ps_exec_guard)
        net.add_input_arc(f"PS_lane_{lane}_input", f"PS_exec_lane_{lane}", _p("frag"))
        _ro(net, "PS_cbuffer", f"PS_exec_lane_{lane}", "cbuf")
        _ro(net, "PS_shader", f"PS_exec_lane_{lane}", "ps_shader")
        net.add_output_arc(f"PS_lane_{lane}_output", f"PS_exec_lane_{lane}",
                           lambda b: Multiset.of(ps_transform(b["frag"])))

        net.add_transition(f"PS_collect_lane_{lane}")
        net.add_input_arc(f"PS_lane_{lane}_output", f"PS_collect_lane_{lane}", _p("px"))
        net.add_output_arc("PS_output", f"PS_collect_lane_{lane}", _p("px"))
        net.add_output_arc(f"PS_lane_{lane}_slot", f"PS_collect_lane_{lane}",
                           lambda _b, _l=lane: Multiset.of(LaneSlot(lane=_l)))

        req_name = f"PS_mem_request_lane_{lane}"
        net.add_transition(req_name)
        _ro(net, f"PS_lane_{lane}_input", req_name, "frag")
        net.add_input_arc(f"PS_mem_req_slot_{lane}", req_name,
                          lambda _b, _l=lane: Multiset.of(LaneSlot(lane=_l)))
        net.add_output_arc("PS_mem_wait", req_name,
                           lambda _b, _l=lane: Multiset.of(LaneSlot(lane=_l)))

        acq_name = f"PS_mem_acquire_lane_{lane}"
        net.add_transition(acq_name)
        net.add_input_arc("PS_mem_wait", acq_name,
                          lambda _b, _l=lane: Multiset.of(LaneSlot(lane=_l)))
        net.add_input_arc("MEM_free", acq_name, _free_expr)
        net.add_output_arc("PS_mem_acquired", acq_name,
                           lambda _b, _l=lane: Multiset.of(LaneSlot(lane=_l)))

        rel_name = f"PS_mem_release_lane_{lane}"
        net.add_transition(rel_name)
        net.add_input_arc("PS_mem_acquired", rel_name,
                          lambda _b, _l=lane: Multiset.of(LaneSlot(lane=_l)))
        net.add_output_arc("MEM_free", rel_name, _free_expr)
        net.add_output_arc(f"PS_mem_req_slot_{lane}", rel_name,
                           lambda _b, _l=lane: Multiset.of(LaneSlot(lane=_l)))

    net.add_transition("PS_sample_texture_0")
    _ro(net, "PS_input", "PS_sample_texture_0", "frag")
    _ro(net, "PS_sampler", "PS_sample_texture_0", "samp")
    _ro(net, "PS_srv", "PS_sample_texture_0", "srv")

    net.add_transition("PS_sample_texture_1")
    _ro(net, "PS_input", "PS_sample_texture_1", "frag")
    _ro(net, "PS_sampler", "PS_sample_texture_1", "samp")
    _ro(net, "PS_srv", "PS_sample_texture_1", "srv")

    net.add_transition("PS_uav_write")
    _ro(net, "PS_input", "PS_uav_write", "frag")
    _ro(net, "PS_uav", "PS_uav_write", "uav")

    net.add_transition("PS_discard")
    net.add_input_arc("PS_input", "PS_discard", _p("frag"))
    _ro(net, "PS_discard_signal", "PS_discard", "sig")

    return net
