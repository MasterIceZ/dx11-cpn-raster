from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from models.dx11_full_pipeline import build_full_pipeline

# Number of lanes to *draw*.  The modelled stage has four; drawing two and
# annotating "x4" keeps the figure legible at one-column width without losing
# information, since the lanes are structurally identical.
DRAW_LANES = int(sys.argv[1]) if len(sys.argv) > 1 else 2
MODEL_LANES = 4

OUT = (Path(__file__).parent.parent / "output" /
       f"vertex_shader_net_{DRAW_LANES}lane.png")


VS_PLACES = ["VS_input", "VS_shader", "VS_cbuffer"] + [
    f"VS_lane_{l}_{io_}" for l in range(DRAW_LANES) for io_ in ("input", "output")
] + ["VS_output"]
VS_TRANSITIONS = [
    f"VS_{op}_lane_{l}" for l in range(DRAW_LANES)
    for op in ("dispatch", "exec", "collect")
]

VS_NODES = set(VS_PLACES + VS_TRANSITIONS)

net = build_full_pipeline()


G = nx.DiGraph()
for n in VS_PLACES:
    G.add_node(n, kind="place")
for n in VS_TRANSITIONS:
    G.add_node(n, kind="transition")

for arc in net.arcs:
    p, t = arc.place.name, arc.transition.name
    if p in VS_NODES and t in VS_NODES:
        if arc.direction == "in":
            G.add_edge(p, t)
        else:
            G.add_edge(t, p)





PLACE_COLOR = "#ffe0aa"
TRANS_COLOR = "#ffffaa"
RESOURCE_COLOR = "#aaddff"


pos: dict[str, tuple[float, float]] = {}


pos["VS_input"]  = (0, 0)


pos["VS_shader"]  = (1.5,  0.7)
pos["VS_cbuffer"] = (1.5, -0.7)


lane_y = {l: (DRAW_LANES - 1 - 2 * l) * 1.5 for l in range(DRAW_LANES)}
for lane, y in lane_y.items():
    pos[f"VS_lane_{lane}_input"]    = (2.5,  y)
    pos[f"VS_dispatch_lane_{lane}"] = (3.5,  y)
    pos[f"VS_lane_{lane}_output"]   = (5.5,  y)
    pos[f"VS_exec_lane_{lane}"]     = (4.5,  y)
    pos[f"VS_collect_lane_{lane}"]  = (6.5,  y)


pos["VS_output"] = (8, 0)


node_colors = []
node_shapes_place = []
node_shapes_trans = []

place_nodes = [n for n in G.nodes if G.nodes[n]["kind"] == "place"]
trans_nodes = [n for n in G.nodes if G.nodes[n]["kind"] == "transition"]


resource = {"VS_shader", "VS_cbuffer"}

fig, ax = plt.subplots(figsize=(16, 2 + 3 * DRAW_LANES))
ax.set_aspect("equal")


pc = [RESOURCE_COLOR if n in resource else PLACE_COLOR for n in place_nodes]
nx.draw_networkx_nodes(G, pos, nodelist=place_nodes, ax=ax,
                       node_shape="o", node_size=9000,
                       node_color=pc, edgecolors="#333333", linewidths=2.2)
nx.draw_networkx_nodes(G, pos, nodelist=trans_nodes, ax=ax,
                       node_shape="s", node_size=7000,
                       node_color=TRANS_COLOR, edgecolors="#333333", linewidths=2.2)


nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                       arrowstyle="-|>", arrowsize=32, edge_color="#333333", width=2.4,
                       min_source_margin=50, min_target_margin=50,
                       connectionstyle="arc3,rad=0.05")


short = {
    "VS_input": "VS_input",
    "VS_output": "VS_output",
    "VS_shader": "VS_shader",
    "VS_cbuffer": "VS_cbuffer",
}
for lane in range(DRAW_LANES):
    short[f"VS_lane_{lane}_input"]    = f"lane_{lane}\ninput"
    short[f"VS_lane_{lane}_output"]   = f"lane_{lane}\noutput"
    short[f"VS_dispatch_lane_{lane}"] = f"dispatch\n{lane}"
    short[f"VS_exec_lane_{lane}"]     = f"exec\n{lane}"
    short[f"VS_collect_lane_{lane}"]  = f"collect\n{lane}"

nx.draw_networkx_labels(G, pos, labels=short, ax=ax, font_size=17)


legend = [
    mpatches.Patch(facecolor=PLACE_COLOR,    edgecolor="#333", label="Buffer place (circle)"),
    mpatches.Patch(facecolor=RESOURCE_COLOR, edgecolor="#333", label="Resource place (circle)"),
    mpatches.Patch(facecolor=TRANS_COLOR,    edgecolor="#333", label="Transition (square)"),
]
ax.legend(handles=legend, loc="center", fontsize=16, framealpha=0.95,
          borderpad=0.4, handlelength=1.2)

# No title: the LaTeX caption carries it, and the extra band of whitespace is
# expensive in a one-column figure.
ax.margins(0.06)
ax.axis("off")
plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")
