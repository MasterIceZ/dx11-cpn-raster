from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import graphviz

from models.dx11_full_pipeline import build_full_pipeline

OUT_DIR = Path(__file__).parent.parent / "output" / "subnets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


PLACE_COLOR     = "#ffe0aa"
RESOURCE_COLOR  = "#aaddff"
SLOT_COLOR      = "#aaffbb"
TRANS_COLOR     = "#ffffaa"
INTERFACE_COLOR = "#ff8888"
EDGE_COLOR      = "#333333"

STAGE_PREFIXES: dict[str, list[str]] = {
    "IA":   ["IA"],
    "VS":   ["VS"],
    "HS":   ["HS"],
    "Tess": ["Tess"],
    "DS":   ["DS"],
    "GS":   ["GS"],
    "SO":   ["SO"],
    "RS":   ["RS"],
    "PS":   ["PS"],
    "OM":   ["OM"],
}

STAGE_TITLES = {
    "IA":   "Input Assembler (IA)",
    "VS":   "Vertex Shader (VS) — 4 lanes",
    "HS":   "Hull Shader (HS)",
    "Tess": "Tessellator",
    "DS":   "Domain Shader (DS)",
    "GS":   "Geometry Shader (GS)",
    "SO":   "Stream Output (SO)",
    "RS":   "Rasterizer (RS)",
    "PS":   "Pixel Shader (PS) — 4 lanes",
    "OM":   "Output Merger (OM)",
}

RESOURCE_KEYWORDS = {"shader", "cbuffer", "sampler", "srv", "uav",
                     "blend_state", "depth_stencil_state", "dsv",
                     "viewport", "scissor", "rasterizer_state",
                     "coverage_mask", "fill_mode", "line_aa_state",
                     "msaa_state", "cull_mode", "depth_clip_enable",
                     "conservative_raster", "domain_type", "max_tess_factor",
                     "output_control_pts", "fork_phase", "join_phase",
                     "instance_count", "max_output_count", "output_topology",
                     "patch_size", "indirect_args", "strip_cut",
                     "write_offset", "rt_slot",
                     "partitioning", "wind_order", "topology_out",
                     "input_layout", "topology", "instance_data",
                     "vertex_id", "draw_cmd"}

def is_resource(name: str) -> bool:
    low = name.lower()
    return any(kw in low for kw in RESOURCE_KEYWORDS)

def is_lane_slot(name: str) -> bool:
    return "_lane_" in name and name.endswith("_slot")

def is_interface(name: str, prefixes: list[str]) -> bool:
    """Stage-boundary buffer: the place a neighboring stage feeds into or reads
    from (e.g. VS_input, VS_output), as opposed to internal per-lane buffers
    such as VS_lane_0_input."""
    return any(name in (f"{pfx}_input", f"{pfx}_output") for pfx in prefixes)

def node_color(name: str, kind: str, prefixes: list[str] = ()) -> str:
    if kind == "transition":
        return TRANS_COLOR
    if is_interface(name, prefixes):
        return INTERFACE_COLOR
    if is_lane_slot(name):
        return SLOT_COLOR
    if is_resource(name):
        return RESOURCE_COLOR
    return PLACE_COLOR

def short_label(name: str, prefixes: list[str]) -> str:
    label = name
    for pfx in prefixes:
        if label.startswith(pfx + "_"):
            label = label[len(pfx) + 1:]
            break

    return label.replace("_", "\n")

def build_dot(net, prefixes: list[str], title: str,
              engine: str = "dot") -> graphviz.Digraph:
    prefix_set = set(prefixes)

    def belongs(name: str) -> bool:
        return name.split("_")[0] in prefix_set

    g = graphviz.Digraph(
        name=title,
        engine=engine,
        graph_attr={
            "label": title,
            "labelloc": "t",
            "fontsize": "18",
            "fontname": "Helvetica",
            "bgcolor": "white",
            "pad": "0.5",
            "nodesep": "0.6",
            "ranksep": "0.8",
            "splines": "line",
        },
    )


    place_names = {name for name in net.places if belongs(name)}
    for name in sorted(place_names):
        color = node_color(name, "place", prefixes)
        label = short_label(name, prefixes)
        interface = is_interface(name, prefixes)
        g.node(name,
               label=label,
               shape="circle",
               style="filled",
               fillcolor=color,
               color=EDGE_COLOR,
               peripheries="2" if interface else "1",
               penwidth="2.0" if interface else "1.0",
               fontsize="9",
               fontname="Helvetica",
               width="0.6",
               fixedsize="false")


    trans_names = {name for name in net.transitions if belongs(name)}
    for name in sorted(trans_names):
        label = short_label(name, prefixes)
        g.node(name,
               label=label,
               shape="rectangle",
               style="filled",
               fillcolor=TRANS_COLOR,
               color=EDGE_COLOR,
               fontsize="9",
               fontname="Helvetica",
               width="0.7",
               height="0.4",
               fixedsize="false")


    nodes = place_names | trans_names
    for arc in net.arcs:
        p, t = arc.place.name, arc.transition.name
        if p in nodes and t in nodes:
            if arc.direction == "in":
                g.edge(p, t, color=EDGE_COLOR, arrowsize="0.7")
            else:
                g.edge(t, p, color=EDGE_COLOR, arrowsize="0.7")

    if any(is_interface(name, prefixes) for name in place_names):
        with g.subgraph(name="cluster_legend") as legend:
            legend.attr(label="Legend", fontsize="10", fontname="Helvetica",
                        style="rounded", color="#999999", bgcolor="#f7f7f7")
            legend.node("legend_interface", label="Stage I/O\ninterface", shape="circle",
                        style="filled", fillcolor=INTERFACE_COLOR, peripheries="2",
                        penwidth="2.0", fontsize="8", width="0.55", fixedsize="false")
            legend.node("legend_buffer", label="Buffer\nplace", shape="circle",
                        style="filled", fillcolor=PLACE_COLOR, fontsize="8",
                        width="0.55", fixedsize="false")
            legend.node("legend_resource", label="Resource\nplace", shape="circle",
                        style="filled", fillcolor=RESOURCE_COLOR, fontsize="8",
                        width="0.55", fixedsize="false")
            legend.node("legend_slot", label="Lane-slot\nsemaphore", shape="circle",
                        style="filled", fillcolor=SLOT_COLOR, fontsize="8",
                        width="0.55", fixedsize="false")
            legend.node("legend_trans", label="Transition", shape="rectangle",
                        style="filled", fillcolor=TRANS_COLOR, fontsize="8",
                        width="0.6", height="0.35", fixedsize="false")
            legend.edge("legend_interface", "legend_buffer", style="invis")
            legend.edge("legend_buffer", "legend_resource", style="invis")
            legend.edge("legend_resource", "legend_slot", style="invis")
            legend.edge("legend_slot", "legend_trans", style="invis")

    return g

def render_stage(net, stage: str, prefixes: list[str]) -> Path:
    title = STAGE_TITLES[stage]
    g = build_dot(net, prefixes, title, engine="dot")

    out_stem = str(OUT_DIR / f"{stage.lower()}_subnet")
    g.render(out_stem, format="png", cleanup=True)
    out = Path(out_stem + ".png")

    n_places = sum(1 for n in net.places if n.split("_")[0] in set(prefixes))
    n_trans  = sum(1 for n in net.transitions if n.split("_")[0] in set(prefixes))
    print(f"  Saved: {out.name}  ({n_places}P + {n_trans}T)")
    return out

def render_full_pipeline(net) -> Path:
    g = graphviz.Digraph(
        name="full_pipeline",
        engine="dot",
        graph_attr={
            "label": "Full DX11 Pipeline CPN (133 places, 126 transitions)",
            "labelloc": "t",
            "fontsize": "20",
            "fontname": "Helvetica",
            "bgcolor": "white",
            "pad": "1.0",
            "rankdir": "LR",
            "splines": "line",
            "nodesep": "0.35",
            "ranksep": "1.0",
        },
    )

    for name in sorted(net.places):
        color = node_color(name, "place", [name.split("_")[0]])
        g.node(name,
               label=name.replace("_", "\n"),
               shape="circle",
               style="filled",
               fillcolor=color,
               color=EDGE_COLOR,
               fontsize="6",
               fontname="Helvetica",
               width="0.45",
               fixedsize="false")

    for name in sorted(net.transitions):
        g.node(name,
               label=name.replace("_", "\n"),
               shape="rectangle",
               style="filled",
               fillcolor=TRANS_COLOR,
               color=EDGE_COLOR,
               fontsize="6",
               fontname="Helvetica",
               width="0.5",
               height="0.3",
               fixedsize="false")

    for arc in net.arcs:
        p, t = arc.place.name, arc.transition.name
        if arc.direction == "in":
            g.edge(p, t, color=EDGE_COLOR, arrowsize="0.5")
        else:
            g.edge(t, p, color=EDGE_COLOR, arrowsize="0.5")

    out_stem = str(OUT_DIR / "full_pipeline")
    g.render(out_stem, format="png", cleanup=True)
    out = Path(out_stem + ".png")
    print(f"  Saved: {out.name}  ({len(net.places)}P + {len(net.transitions)}T)")
    return out

def main() -> None:
    print("Building full pipeline model…")
    net = build_full_pipeline()
    print("Rendering subnets (graphviz dot)…")
    for stage, prefixes in STAGE_PREFIXES.items():
        render_stage(net, stage, prefixes)
    print("Rendering full pipeline (graphviz sfdp)…")
    render_full_pipeline(net)
    print("Done.")

if __name__ == "__main__":
    main()
