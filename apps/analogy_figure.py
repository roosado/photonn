"""Static counterpart to the browser correspondence figure (Phase 3 docs).

Draws the same argument ``apps/web/analogy.js`` makes interactively -- the shared
``[phase] -> [mixing] -> ... -> |E|^2`` skeleton, and the reach lightcone that
explains why one machine is six layers deep and the other thirty-six -- as a
single static plate for ``docs/phase3_mesh.md``.

Both read the *same* generated geometry (``apps/web/analogy_geom.js``, written by
:mod:`apps.export_analogy_web`), so the figure and the widget cannot disagree.

    python -m apps.analogy_figure
"""
from __future__ import annotations

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOM_JS = os.path.join(_REPO, "apps", "web", "analogy_geom.js")
OUT_PNG = os.path.join(_REPO, "docs", "figures", "phase3_correspondence.png")

PHASE_C = "#2f7d8c"      # trainable phase, in both machines
MIX_C = "#c9701f"        # fixed mixing hardware, in both machines
INK = "#1b2430"
MUTED = "#6b7789"


def load_geom(path: str = GEOM_JS) -> dict:
    """Parse the generated geometry bundle. It is a JSON object inside a JS shim."""
    text = open(path, encoding="utf-8").read()
    body = re.search(r"var G = (\{.*?\});\n", text, re.S)
    if body is None:
        raise ValueError(f"{path} does not look like a generated analogy_geom bundle.")
    return json.loads(body.group(1))


def _interleave(kind_a, n_a, kind_b, n_b, start):
    """Alternate two stripe kinds with exact counts, beginning with ``start``."""
    out, a, b, turn = [], 0, 0, start
    while a < n_a or b < n_b:
        if turn == kind_a and a < n_a:
            out.append(kind_a); a += 1; turn = kind_b
        elif turn == kind_b and b < n_b:
            out.append(kind_b); b += 1; turn = kind_a
        elif a < n_a:
            out.append(kind_a); a += 1
        else:
            out.append(kind_b); b += 1
    return out


def draw_skeleton(ax, geom):
    """Both machines as the same alternating stripe rhythm, at their true depths."""
    d, m = geom["d2nn"], geom["mesh"]
    tracks = [
        (_interleave("mix", d["hops"], "phase", d["n_layers"], "mix"),
         "D²NN\nfree space", f"{d['n_layers']} phase layers  ·  {d['hops']} mixing layers"),
        (_interleave("phase", m["n_columns"], "mix", m["n_columns"], "phase"),
         "MZI mesh\nchip", f"{m['n_columns']} phase layers  ·  {m['n_columns']} mixing layers"),
    ]
    for row, (skel, name, depth) in enumerate(tracks):
        y = -row * 1.05
        w = 1.0 / len(skel)
        for i, kind in enumerate(skel):
            ax.add_patch(Rectangle((i * w, y), w, 0.55, lw=0,
                                   facecolor=PHASE_C if kind == "phase" else MIX_C,
                                   alpha=0.88))
        ax.add_patch(Rectangle((1.015, y + 0.07), 0.035, 0.41, lw=1.0,
                               edgecolor=INK, facecolor="none"))
        ax.text(1.0325, y + 0.275, "$|E|^2$", ha="center", va="center", fontsize=7, color=INK)
        ax.text(-0.015, y + 0.275, name, ha="right", va="center", fontsize=8.5, color=INK)
        ax.text(0.0, y - 0.1, depth, ha="left", va="center", fontsize=7.5, color=MUTED)

    ax.text(0.5, -0.36, "same rhythm, different depth", ha="center", va="center",
            fontsize=7.5, color=MUTED, style="italic")
    ax.set_xlim(-0.30, 1.10)
    ax.set_ylim(-1.30, 0.72)
    ax.axis("off")
    ax.set_title("One skeleton, two machines — only the phases are trainable",
                 fontsize=10, color=INK, pad=4)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=PHASE_C, alpha=.88, label="trainable phase"),
                       Rectangle((0, 0), 1, 1, facecolor=MIX_C, alpha=.88, label="fixed mixing")],
              loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=8)


def draw_d2nn_cone(ax, geom):
    """Reach of the diffractive stack: a lightcone from the worst-case input pixel."""
    d = geom["d2nn"]
    n, hops, reach = d["n"], d["hops"], d["reach_px_per_hop"]
    src = d["input_window"][1]

    ax.add_patch(Polygon([(src, 0), (src + hops * reach, hops), (src - hops * reach, hops)],
                         closed=True, facecolor=MIX_C, alpha=0.16, lw=0))
    for edge in (-1, 1):
        ax.plot([src, src + edge * hops * reach], [0, hops], color=MIX_C, lw=1.3)

    for k in range(1, hops):
        ax.axhline(k, color=PHASE_C, lw=1.0, alpha=0.8)
    ax.plot(d["input_window"], [0, 0], color=INK, lw=3, solid_capstyle="butt")
    ax.plot([src], [0], "o", color=MIX_C, ms=4)

    for x0, x1 in sorted({(r[2], r[3] - 1) for r in d["regions"]}):
        ax.plot([x0, x1], [hops, hops], color=INK, lw=4, alpha=0.6, solid_capstyle="butt")

    edge = src - d["reach_px_total"]
    ax.axvline(edge, color="#c14a34", lw=1.2, ls="--")
    ax.annotate(f"cone edge {edge:.1f} px\ndetector starts {d['detector_x'][0]} px\n"
                f"→ {d['margin_px']:.2f} px of margin",
                xy=(edge, hops), xytext=(edge + 4, hops - 2.5), fontsize=7.5, color="#c14a34",
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#c14a34",
                          lw=0.6, alpha=0.94),
                arrowprops=dict(arrowstyle="->", color="#c14a34", lw=0.8,
                                connectionstyle="arc3,rad=-0.2"))

    ax.text(d["input_window"][0], -0.22, "input window", fontsize=7.5, color=MUTED)
    ax.text(n, hops + 0.34, "ten detectors", fontsize=7.5, color=MUTED, ha="right")
    ax.set_xlim(0, n - 1)
    ax.set_ylim(hops + 0.55, -0.55)
    ax.set_xlabel("pixel across the field", fontsize=8.5)
    ax.set_ylabel("free-space hop", fontsize=8.5)
    ax.set_yticks(range(hops + 1))
    ax.tick_params(labelsize=7.5)
    ax.set_title(f"Diffraction: {reach:.1f} px per hop, {hops} hops → "
                 f"{d['reach_px_total']:.1f} px",
                 fontsize=9.5, color=INK, pad=6)


def draw_mesh_cone(ax, geom):
    """Reach of the mesh: one mode per coupler column, drawn on the real schedule."""
    m = geom["mesh"]
    modes, cols = m["n_modes"], m["n_columns"]
    src = modes - 1

    for mode in range(modes):
        ax.plot([0, cols], [mode, mode], color="0.86", lw=0.6, zorder=0)
    for c, tops in enumerate(m["columns"]):
        for top in tops:
            ax.add_patch(Rectangle((c + 0.28, top - 0.04), 0.44, 1.08, lw=0,
                                   facecolor=PHASE_C, alpha=0.5, zorder=2))

    # Cone over the topology, so the reach reads against the brick pattern.
    ax.add_patch(Polygon([(0, src), (cols, src - cols), (cols, min(modes - 1, src + cols))],
                         closed=True, facecolor=MIX_C, alpha=0.26, lw=0, zorder=3))
    ax.plot([0, cols], [src, src - cols], color=MIX_C, lw=1.6, zorder=4)
    ax.plot([0], [src], "o", color=MIX_C, ms=4, zorder=4)

    ax.plot([cols + 0.35] * 2, [0, 9], color=INK, lw=4, alpha=0.6, solid_capstyle="butt")
    ax.text(cols + 0.7, 4.5, "10 output\nmodes", fontsize=7.5, color=MUTED, va="center")
    ax.set_xlim(-0.4, cols + 3.2)
    ax.set_ylim(modes - 0.4, -0.6)
    ax.set_xlabel("coupler column", fontsize=8.5)
    ax.set_ylabel("waveguide mode", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_title(f"Couplers: 1 mode per column, {cols} columns → all {modes} modes",
                 fontsize=9.5, color=INK, pad=6)


def render(geom=None):
    geom = geom or load_geom()
    fig = plt.figure(figsize=(11.5, 7.4), constrained_layout=True)
    fig.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.72, 1.0])

    draw_skeleton(fig.add_subplot(gs[0, :]), geom)
    draw_d2nn_cone(fig.add_subplot(gs[1, 0]), geom)
    draw_mesh_cone(fig.add_subplot(gs[1, 1]), geom)

    d = geom["d2nn"]
    fig.suptitle("Free space ↔ chip: the same alternating machine, two reaches per layer",
                 fontsize=12, color=INK)
    fig.text(0.5, -0.005,
             f"Reach per layer is the whole difference. Diffraction gives "
             f"{d['reach_px_per_hop']:.1f} px for free but unsteerable; a coupler gives exactly one "
             f"mode, individually steerable. Both designs land just inside full connectivity — "
             f"the D²NN by {d['margin_px']:.2f} px of geometry, the mesh by the Clements bound.",
             ha="center", va="top", fontsize=8, color=MUTED, wrap=True)
    return fig


def main():
    fig = render()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PNG} ({os.path.getsize(OUT_PNG) // 1024} KB)")


if __name__ == "__main__":
    main()
