
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pipeline.petri.net import Binding

@dataclass(frozen=True)
class Vertex:

    vertex_id: int
    clip_pos: tuple[float, float, float, float]
    instance_id: int = 0
    attributes: tuple[float, ...] = ()

@dataclass(frozen=True)
class Primitive:

    vertices: tuple[Vertex, ...]
    topology: Literal["triangle_list", "line_list", "point_list"] = "triangle_list"
    instance_id: int = 0

    @property
    def expected_vertex_count(self) -> int:
        return {"triangle_list": 3, "line_list": 2, "point_list": 1}[self.topology]

@dataclass(frozen=True)
class Fragment:

    pixel_x: int
    pixel_y: int
    depth: float
    attributes: tuple[float, ...] = ()

@dataclass(frozen=True)
class Pixel:

    pixel_x: int
    pixel_y: int
    color: tuple[float, float, float, float]
    depth: float
    output_slot: int = 0

@dataclass(frozen=True)
class RenderTarget:

    slot: int
    width: int
    height: int
    format: str = "DXGI_FORMAT_R8G8B8A8_UNORM"


DX11_TOKEN_TYPES: tuple[type, ...] = (
    Vertex, Primitive, Fragment, Pixel, RenderTarget
)





@dataclass
class ColorSet:

    name: str
    token_type: type
    validator: Callable[[Any], bool] | None = None

    def contains(self, token: Any) -> bool:
        if not isinstance(token, self.token_type):
            return False
        return self.validator(token) if self.validator is not None else True

    def validate(self, token: Any) -> None:
        if not self.contains(token):
            raise ValueError(
                f"Token {token!r} does not belong to color set '{self.name}'"
            )

    def __repr__(self) -> str:
        return f"ColorSet({self.name!r}, {self.token_type.__name__})"



VERTEX_CS = ColorSet(
    "Vertex",
    Vertex,
    validator=lambda v: (
        len(v.clip_pos) == 4
        and all(math.isfinite(c) for c in v.clip_pos)
    ),
)

PRIMITIVE_CS = ColorSet(
    "Primitive",
    Primitive,
    validator=lambda p: len(p.vertices) == p.expected_vertex_count,
)

FRAGMENT_CS = ColorSet(
    "Fragment",
    Fragment,
    validator=lambda f: 0.0 <= f.depth <= 1.0,
)

PIXEL_CS = ColorSet(
    "Pixel",
    Pixel,
    validator=lambda px: (
        0.0 <= px.depth <= 1.0
        and all(0.0 <= c <= 1.0 for c in px.color)
        and 0 <= px.output_slot <= 7
    ),
)

RENDER_TARGET_CS = ColorSet(
    "RenderTarget",
    RenderTarget,
    validator=lambda rt: (
        -1 <= rt.slot <= 7
        and rt.width > 0
        and rt.height > 0
    ),
)


COLOR_SETS: dict[type, ColorSet] = {
    Vertex:       VERTEX_CS,
    Primitive:    PRIMITIVE_CS,
    Fragment:     FRAGMENT_CS,
    Pixel:        PIXEL_CS,
    RenderTarget: RENDER_TARGET_CS,
}





class GuardFunction:

    def __init__(self, predicate: Callable[[Binding], bool], name: str = "") -> None:
        self._predicate = predicate
        self.name = name or getattr(predicate, "__name__", "<guard>")

    def __call__(self, binding: Binding) -> bool:
        return self._predicate(binding)

    def __and__(self, other: "GuardFunction") -> "GuardFunction":
        return GuardFunction(
            lambda b: self(b) and other(b),
            f"({self.name} & {other.name})",
        )

    def __or__(self, other: "GuardFunction") -> "GuardFunction":
        return GuardFunction(
            lambda b: self(b) or other(b),
            f"({self.name} | {other.name})",
        )

    def __invert__(self) -> "GuardFunction":
        return GuardFunction(lambda b: not self(b), f"~{self.name}")

    def __repr__(self) -> str:
        return f"GuardFunction({self.name!r})"

def guard(name: str = "") -> Callable[[Callable[[Binding], bool]], GuardFunction]:
    def decorator(fn: Callable[[Binding], bool]) -> GuardFunction:
        return GuardFunction(fn, name or fn.__name__)
    return decorator













@guard("vertex_clip_w_positive")
def vertex_clip_w_positive(b: Binding) -> bool:
    v: Vertex = b["v"]
    return v.clip_pos[3] > 0.0

@guard("vertex_pos_finite")
def vertex_pos_finite(b: Binding) -> bool:
    v: Vertex = b["v"]
    return all(math.isfinite(c) for c in v.clip_pos)


vertex_valid: GuardFunction = vertex_pos_finite & vertex_clip_w_positive
vertex_valid.name = "vertex_valid"



@guard("primitive_vertex_count_valid")
def primitive_vertex_count_valid(b: Binding) -> bool:
    prim: Primitive = b["prim"]
    return len(prim.vertices) == prim.expected_vertex_count

@guard("primitive_vertices_finite")
def primitive_vertices_finite(b: Binding) -> bool:
    prim: Primitive = b["prim"]
    return all(
        math.isfinite(c)
        for v in prim.vertices
        for c in v.clip_pos
    )

@guard("primitive_not_degenerate")
def primitive_not_degenerate(b: Binding) -> bool:
    prim: Primitive = b["prim"]
    if prim.topology != "triangle_list":
        return True
    xys = [(v.clip_pos[0] / v.clip_pos[3], v.clip_pos[1] / v.clip_pos[3])
           for v in prim.vertices]
    return len(set(xys)) > 1

primitive_valid: GuardFunction = (
    primitive_vertex_count_valid & primitive_vertices_finite
)
primitive_valid.name = "primitive_valid"



@guard("fragment_depth_in_range")
def fragment_depth_in_range(b: Binding) -> bool:
    frag: Fragment = b["frag"]
    return 0.0 <= frag.depth <= 1.0

def fragment_in_viewport(width: int, height: int) -> GuardFunction:
    return GuardFunction(
        lambda b: (
            0 <= b["frag"].pixel_x < width
            and 0 <= b["frag"].pixel_y < height
        ),
        f"fragment_in_viewport({width}x{height})",
    )

fragment_valid: GuardFunction = fragment_depth_in_range
fragment_valid.name = "fragment_valid"



@guard("pixel_depth_in_range")
def pixel_depth_in_range(b: Binding) -> bool:
    px: Pixel = b["px"]
    return 0.0 <= px.depth <= 1.0

@guard("pixel_color_normalized")
def pixel_color_normalized(b: Binding) -> bool:
    px: Pixel = b["px"]
    return all(0.0 <= c <= 1.0 for c in px.color)

@guard("pixel_slot_valid")
def pixel_slot_valid(b: Binding) -> bool:
    px: Pixel = b["px"]
    return 0 <= px.output_slot <= 7

pixel_valid: GuardFunction = pixel_depth_in_range & pixel_slot_valid
pixel_valid.name = "pixel_valid"



@guard("rt_slot_valid")
def rt_slot_valid(b: Binding) -> bool:
    rt: RenderTarget = b["rt"]
    return -1 <= rt.slot <= 7

@guard("rt_dimensions_valid")
def rt_dimensions_valid(b: Binding) -> bool:
    rt: RenderTarget = b["rt"]
    return rt.width > 0 and rt.height > 0

render_target_valid: GuardFunction = rt_slot_valid & rt_dimensions_valid
render_target_valid.name = "render_target_valid"



def pixel_in_rt_bounds(
    px_key: str = "px",
    rt_key: str = "rt",
) -> GuardFunction:
    return GuardFunction(
        lambda b: (
            0 <= b[px_key].pixel_x < b[rt_key].width
            and 0 <= b[px_key].pixel_y < b[rt_key].height
        ),
        f"pixel_in_rt_bounds({px_key}, {rt_key})",
    )

def pixel_slot_matches_rt(
    px_key: str = "px",
    rt_key: str = "rt",
) -> GuardFunction:
    return GuardFunction(
        lambda b: b[px_key].output_slot == b[rt_key].slot,
        f"pixel_slot_matches_rt({px_key}, {rt_key})",
    )

@guard("depth_write_not_masked")
def depth_write_not_masked(b: Binding) -> bool:
    return bool(b.get("depth_write_mask", True))

def depth_test_passes(comparison: Literal["less", "less_equal", "always"] = "less") -> GuardFunction:
    ops: dict[str, Callable[[float, float], bool]] = {
        "less":       lambda incoming, cur: incoming < cur,
        "less_equal": lambda incoming, cur: incoming <= cur,
        "always":     lambda _i, _c: True,
    }
    op = ops[comparison]
    return GuardFunction(
        lambda b: op(b["px"].depth, b.get("current_depth", 1.0)),
        f"depth_test_passes({comparison})",
    )





def all_guards(*guards: GuardFunction) -> GuardFunction:
    if not guards:
        return GuardFunction(lambda _: True, "always_true")
    result = guards[0]
    for g in guards[1:]:
        result = result & g
    return result
