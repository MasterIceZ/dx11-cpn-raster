
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from pipeline.petri.colored import (
    Binding,
    GuardFunction,
    Vertex,
    guard,
)
from pipeline.petri.net import Multiset, PetriNet





@dataclass(frozen=True)
class ConstantBuffer:

    slot: int
    size: int = 1
    data: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not (0 <= self.slot <= 13):
            raise ValueError(
                f"VS cbuffer slot must be in [0, 13], got {self.slot}"
            )
        if not (1 <= self.size <= 4096):
            raise ValueError(
                f"cbuffer size must be in [1, 4096] vec4 constants, got {self.size}"
            )

@dataclass(frozen=True)
class ShaderBinding:

    stage: str
    name: str = "<anonymous>"
    valid: bool = True





@guard("vs_shader_bound")
def vs_shader_bound(b: Binding) -> bool:
    shader: ShaderBinding | None = b.get("vs_shader")
    return shader is not None and shader.stage == "VS" and shader.valid

@guard("vs_cbuffer_slot_valid")
def vs_cbuffer_slot_valid(b: Binding) -> bool:
    cbuf: ConstantBuffer | None = b.get("cbuf")
    return cbuf is not None and 0 <= cbuf.slot <= 13

@guard("vs_output_sv_position_valid")
def vs_output_sv_position_valid(b: Binding) -> bool:
    v_out: Vertex | None = b.get("v_out")
    if v_out is None:
        return False
    x, y, z, w = v_out.clip_pos
    return all(math.isfinite(c) for c in (x, y, z, w)) and w != 0.0


_vs_exec_guard: GuardFunction = (
    vs_shader_bound & vs_cbuffer_slot_valid & vs_output_sv_position_valid
)





def build_subnet(config: dict[str, Any] | None = None) -> PetriNet:
    cfg: dict[str, Any] = config or {}
    max_vertices: int = cfg.get("max_vertices", 65_536)
    shader_name: str = cfg.get("shader_name", "DefaultVS")

    net = PetriNet(name="VS")






    net.add_place(
        name="VS_input",
        color_type=Vertex,
        bound=max_vertices,
    )



    net.add_place(
        name="VS_cbuffer",
        color_type=ConstantBuffer,
        bound=14,
    )




    net.add_place(
        name="VS_shader",
        color_type=ShaderBinding,
        initial=[ShaderBinding(stage="VS", name=shader_name, valid=True)],
        bound=1,
    )


    net.add_place(
        name="VS_output",
        color_type=Vertex,
        bound=max_vertices,
    )





    net.add_transition(name="VS_exec", guard=_vs_exec_guard)






    net.add_input_arc(
        place="VS_input",
        transition="VS_exec",
        expression=lambda b: Multiset.of(b["v"]),
    )


    net.add_input_arc(
        place="VS_cbuffer",
        transition="VS_exec",
        expression=lambda b: Multiset.of(b["cbuf"]),
    )
    net.add_output_arc(
        place="VS_cbuffer",
        transition="VS_exec",
        expression=lambda b: Multiset.of(b["cbuf"]),
    )


    net.add_input_arc(
        place="VS_shader",
        transition="VS_exec",
        expression=lambda b: Multiset.of(b["vs_shader"]),
    )
    net.add_output_arc(
        place="VS_shader",
        transition="VS_exec",
        expression=lambda b: Multiset.of(b["vs_shader"]),
    )



    net.add_output_arc(
        place="VS_output",
        transition="VS_exec",
        expression=lambda b: Multiset.of(b["v_out"]),
    )

    return net


# ---------------------------------------------------------------------------
# SNAKES-based subnet
# ---------------------------------------------------------------------------

def build_snakes_subnet(
    config: dict[str, Any] | None = None,
    initial_vertices: list[Vertex] | None = None,
    initial_cbufs: list[ConstantBuffer] | None = None,
) -> "snakes.nets.PetriNet":
    """Build the VS subnet using the SNAKES library.

    This is a parallel implementation of :func:`build_subnet` that uses
    ``snakes.nets`` for place/transition/arc representation.  Arc expressions
    and guards are written as SNAKES ``Expression`` strings evaluated in a
    context that includes our domain globals.

    Parameters
    ----------
    config:
        Same keys as :func:`build_subnet`.
    initial_vertices:
        Vertex tokens placed in ``VS_input`` at construction time.
    initial_cbufs:
        ConstantBuffer tokens placed in ``VS_cbuffer`` at construction time.

    Returns
    -------
    snakes.nets.PetriNet
        A SNAKES net ready for ``Transition.modes()`` / ``Transition.fire()``
        or :func:`pipeline.petri.snakes_bridge.snakes_to_state_space`.
    """
    import math as _math
    from snakes.nets import (
        Expression,
        PetriNet as SNKNet,
        Place as SNKPlace,
        Transition as SNKTransition,
        Variable,
    )

    cfg = config or {}
    shader_name: str = cfg.get("shader_name", "DefaultVS")

    net = SNKNet("VS")

    # ------------------------------------------------------------------
    # Register globals available in SNAKES arc/guard expressions
    # ------------------------------------------------------------------
    net.globals["math"] = _math
    net.globals["Vertex"] = Vertex

    def _transform(v: Vertex, cbuf: ConstantBuffer) -> Vertex:
        """Simulated VS: scale x by 0.5 (placeholder world→clip transform)."""
        x, y, z, w = v.clip_pos
        return Vertex(
            vertex_id=v.vertex_id,
            clip_pos=(x * 0.5, y, z, w),
            instance_id=v.instance_id,
            attributes=v.attributes,
        )

    net.globals["transform_vertex"] = _transform

    # ------------------------------------------------------------------
    # Initial tokens
    # ------------------------------------------------------------------
    shader_tok = ShaderBinding(stage="VS", name=shader_name, valid=True)
    vertices = list(initial_vertices or [])
    cbufs = list(initial_cbufs or [])

    # ------------------------------------------------------------------
    # Places
    # ------------------------------------------------------------------
    net.add_place(SNKPlace("VS_input", vertices))
    net.add_place(SNKPlace("VS_cbuffer", cbufs))
    net.add_place(SNKPlace("VS_shader", [shader_tok]))
    net.add_place(SNKPlace("VS_output", []))

    # ------------------------------------------------------------------
    # Transition + guard
    # ------------------------------------------------------------------
    guard_expr = Expression(
        'vs_shader.stage == "VS" and vs_shader.valid'
        " and 0 <= cbuf.slot <= 13"
        " and all(math.isfinite(c) for c in transform_vertex(v, cbuf).clip_pos)"
        " and transform_vertex(v, cbuf).clip_pos[3] != 0.0"
    )
    net.add_transition(SNKTransition("VS_exec", guard_expr))

    # ------------------------------------------------------------------
    # Input arcs  (consume one token per variable binding)
    # ------------------------------------------------------------------
    net.add_input("VS_input", "VS_exec", Variable("v"))
    net.add_input("VS_cbuffer", "VS_exec", Variable("cbuf"))
    net.add_input("VS_shader", "VS_exec", Variable("vs_shader"))

    # ------------------------------------------------------------------
    # Output arcs  (cbuffer + shader are read-only; vertex is transformed)
    # ------------------------------------------------------------------
    net.add_output("VS_cbuffer", "VS_exec", Variable("cbuf"))
    net.add_output("VS_shader", "VS_exec", Variable("vs_shader"))
    net.add_output("VS_output", "VS_exec", Expression("transform_vertex(v, cbuf)"))

    return net
