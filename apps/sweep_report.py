"""Side-by-side plate for the optics sweep, plus the widget's data bundle.

Turns ``exports/sweep/optics_sweep.json`` (written by :mod:`apps.sweep_optics`)
into the two committed artefacts the docs and the site need:

* ``docs/figures/optics_sweep.png`` -- one column per optical configuration,
  worst to best, showing *why* each one scores what it scores: the diffraction
  cone against the connectivity requirement, the detector plane it produces, and
  the measured accuracy. Above them, accuracy against reach with the connectivity
  bound and the pure-diffraction floor drawn in.
* ``apps/web/optics_sweep.js`` -- the same numbers for ``apps/web/optics.js``.
  ``exports/`` is gitignored, so this is the only copy that survives a fresh
  clone (same reasoning as ``analogy_geom.js``).

    python -m apps.sweep_report
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon, Rectangle

from apps.sweep_optics import BASE_LAYERS, BASE_Z_MM, ISO_REACH_PX as ISO_TARGET_PX
from photonn.fields import Field
from photonn.propagate import angular_spectrum
from photonn.train import encode_input, load_dataset

_REPO = Path(__file__).resolve().parent.parent
SWEEP_JSON = _REPO / "exports" / "sweep" / "optics_sweep.json"
OUT_PNG = _REPO / "docs" / "figures" / "optics_sweep.png"
OUT_JS = _REPO / "apps" / "web" / "optics_sweep.js"

GRID, DX, WAVELENGTH = 128, 8e-6, 532e-9
INPUT_LO, INPUT_HI = 32, 95        # entrance window, train.embed_input(input_frac=0.5)
REQUIRED_PX = 74.0                 # worst-case input->detector distance (docs/phase3_mesh.md)

BEAM = "#0f9e8f"
FRINGE = "#c9701f"
BAD = "#c14a34"
GOOD = "#2f8f52"
INK = "#141b26"
MUTED = "#6b7789"


def load_sweep(path=SWEEP_JSON) -> dict:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"no sweep results at {path}; run `python -m apps.sweep_optics --arm z` first.")
    return json.loads(path.read_text())


def detector_plane(z_mm: float, layers: int, digit: np.ndarray, masks=None) -> np.ndarray:
    """|E|^2 at the detector plane for one configuration.

    Uses the configuration's own trained masks when the sweep saved them; falls
    back to pure diffraction (masks = identity) otherwise, which still shows the
    thing the column is making a claim about -- how far the light has spread by
    the time it is read.
    """
    f = Field(digit, DX, WAVELENGTH)
    z = z_mm * 1e-3
    for i in range(layers + 1):
        f = angular_spectrum(f, z)
        if masks is not None and i < layers:
            f = Field(f.data * np.exp(1j * masks[i]), DX, WAVELENGTH)
    return np.abs(f.data) ** 2


def masks_for(run, path=SWEEP_JSON):
    p = Path(path).with_name(f"masks_{run['config']}.npy")
    return np.load(p) if p.exists() else None


def draw_cone(ax, run):
    """The connectivity argument: can an edge input pixel reach the far detector?

    Horizontal axis is the grid in pixels; the wedge is how far one edge of the
    entrance window can throw energy over all hops. If it stops short of the
    requirement the configuration is not merely worse, it is *unable* to compute
    the mapping -- which is what the low-z columns are here to show.
    """
    reach = run["reach_px_total"]
    ax.set_xlim(0, GRID)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 64, GRID])
    ax.tick_params(labelsize=6, colors=MUTED, length=2)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.add_patch(Rectangle((INPUT_LO, 0.80), INPUT_HI - INPUT_LO, 0.14,
                           color=MUTED, alpha=0.35, lw=0))
    ax.text(GRID / 2, 0.97, "entrance window", ha="center", va="top",
            fontsize=5.5, color=MUTED)

    # Worst case runs leftward: the input pixel at px 95 must reach the detector
    # pixel at px 21, 74 px away (docs/phase3_mesh.md). Draw that side.
    covered = reach >= REQUIRED_PX
    far = max(INPUT_HI - reach, 0)
    colour = BEAM if covered else BAD
    ax.add_patch(Polygon([(INPUT_HI, 0.80), (far, 0.06), (min(INPUT_HI + reach, GRID), 0.06)],
                         closed=True, color=colour, alpha=0.22, lw=0))
    ax.plot([INPUT_HI, far], [0.80, 0.06], color=colour, lw=1.2)

    need = INPUT_HI - REQUIRED_PX
    ax.plot([need, need], [0.0, 0.42], color=INK, lw=1.0, ls=(0, (2, 1.6)))
    ax.text(need + 2, 0.46, "farthest\ndetector", ha="left", va="bottom",
            fontsize=5.5, color=INK)

    ax.text(GRID - 2, 0.10, f"{reach:.0f} px", fontsize=7, color=colour,
            va="bottom", ha="right", fontweight="bold")


def split_arms(doc):
    """Separate the two experiments; they answer different questions.

    * **z arm** -- mask count fixed at the shipped 5, separation varied. Asks how
      much of the input each detector can see.
    * **iso-reach arm** -- total reach held at the z arm's winner, the z/L split
      varied. Asks how a fixed reach budget is best spent. Every point here has
      identical reach *and* identical wrap error, so the confound that muddies the
      top of the z arm is held exactly constant.
    """
    trained = [r for r in doc.get("runs", []) if r["layers"] > 0]
    iso_reach = round(ISO_TARGET_PX, 1)
    z_arm = sorted((r for r in trained if r["layers"] == BASE_LAYERS),
                   key=lambda r: r["reach_px_total"])
    iso = sorted((r for r in trained if round(r["reach_px_total"], 1) == iso_reach),
                 key=lambda r: r["layers"])
    floor = next((r for r in doc.get("runs", []) if r["layers"] == 0), None)
    return z_arm, iso, floor


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=8, color=INK)
    ax.set_ylabel(ylabel, fontsize=8, color=INK)
    ax.tick_params(labelsize=7, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=9.5, color=INK, pad=7)


def draw_z_arm(ax, z_arm, floor, geom):
    xs = [r["reach_px_total"] for r in z_arm]
    ys = [r["val_acc"] for r in z_arm]
    ax.plot(xs, ys, "-o", color=BEAM, lw=1.8, ms=5, zorder=3)

    ax.axvline(REQUIRED_PX, color=INK, lw=1.1, ls=(0, (3, 2)), zorder=2)
    ax.axvspan(0, REQUIRED_PX, color=BAD, alpha=0.06, lw=0)
    ax.text(REQUIRED_PX - 4, 0.30, "connectivity bound, 74 px", ha="right",
            va="bottom", rotation=90, fontsize=6.5, color=INK)

    if floor:
        ax.axhline(floor["val_acc"], color=MUTED, lw=1.0, ls=":")
        ax.text(max(xs), floor["val_acc"] + 0.014,
                f"pure diffraction, 0 parameters ({floor['val_acc']:.3f})",
                ha="right", va="bottom", fontsize=6.5, color=MUTED)

    for r in z_arm:
        wrap = geom.get((r["z_mm"], r["layers"]), {}).get("logit_error")
        note = f"{r['z_mm']:g} mm"
        if wrap is not None and wrap > 0.02:
            note += "\nwrap-marginal"
            ax.plot([r["reach_px_total"]], [r["val_acc"]], "o", ms=10, mfc="none",
                    mec=FRINGE, mew=1.4, zorder=4)
        # The point sitting on the bound labels below instead, so it clears both
        # the bound line and its right-hand neighbour.
        below = abs(r["reach_px_total"] - REQUIRED_PX) < 12
        ax.annotate(note, (r["reach_px_total"], r["val_acc"]),
                    textcoords="offset points", xytext=(0, -10 if below else 9),
                    va="top" if below else "bottom",
                    ha="center", fontsize=6.5, color=INK)

    ax.set_ylim(0.05, max(ys) + 0.10)
    _style(ax, "total reach over all hops (px)", "validation accuracy",
           "Separation: how much of the input a detector can see")


def draw_iso_arm(ax, iso, z_arm):
    if not iso:
        ax.set_visible(False)
        return
    xs = [r["layers"] for r in iso]
    ys = [r["val_acc"] for r in iso]
    ax.plot(xs, ys, "-o", color=FRINGE, lw=1.8, ms=5, zorder=3)

    shipped = next((r for r in z_arm if r["z_mm"] == BASE_Z_MM), None)
    if shipped:
        ax.axhline(shipped["val_acc"], color=MUTED, lw=1.0, ls=":")
        ax.text(max(xs), shipped["val_acc"] - 0.008,
                f"shipped geometry ({shipped['val_acc']:.4f})",
                ha="right", va="top", fontsize=6.5, color=MUTED)

    for r in iso:
        ax.annotate(f"{r['z_mm']:g} mm\n{r['n_params'] // 1000}k", (r["layers"], r["val_acc"]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=6.5, color=INK)

    ax.set_xticks(xs)
    ax.set_ylim(min(ys) - 0.04, max(ys) + 0.075)
    _style(ax, "trainable phase masks", "validation accuracy",
           f"Same {ISO_TARGET_PX:.0f} px of reach, spent differently")


def build_figure(doc, out=OUT_PNG):
    z_arm, iso, floor = split_arms(doc)
    geom = {(g["z_mm"], g["layers"]): g for g in doc.get("geometry", [])}
    if not z_arm:
        raise SystemExit("sweep results contain no trained configurations yet.")

    # The progression the figure exists to make, in the order a reader walks it:
    # broken -> shipped -> more separation -> more masks -> best.
    shortest = z_arm[0]
    shipped = next((r for r in z_arm if r["z_mm"] == BASE_Z_MM), None)
    best_z = max(z_arm, key=lambda r: r["val_acc"])
    ordered = [shortest, shipped, best_z] + iso[1:]

    seen, columns = set(), []
    for r in ordered:
        if r is not None and r["config"] not in seen:
            seen.add(r["config"])
            columns.append(r)

    ds = load_dataset("mnist", subset=64, split="test")
    k = int(np.argmax(ds.labels == 3))          # one fixed digit for every column
    digit = encode_input(torch.as_tensor(ds.images[k:k + 1], dtype=torch.float32),
                         scheme="both", n=GRID).numpy()[0]

    ncol = len(columns)
    fig = plt.figure(figsize=(2.15 * ncol, 8.0), dpi=200)
    gs = fig.add_gridspec(3, ncol, height_ratios=[1.35, 0.50, 1.0],
                          hspace=0.46, wspace=0.16,
                          left=0.055, right=0.985, top=0.945, bottom=0.055)

    half = max(2, ncol // 2)
    draw_z_arm(fig.add_subplot(gs[0, :half]), z_arm, floor, geom)
    draw_iso_arm(fig.add_subplot(gs[0, half:]), iso, z_arm)

    for j, r in enumerate(columns):
        draw_cone(fig.add_subplot(gs[1, j]), r)

        axp = fig.add_subplot(gs[2, j])
        img = detector_plane(r["z_mm"], r["layers"], digit, masks_for(r))
        axp.imshow(img ** 0.4, cmap="magma", interpolation="nearest")
        axp.set_xticks([]); axp.set_yticks([])
        for s in axp.spines.values():
            s.set_color(MUTED); s.set_linewidth(0.6)

        ok = r["reach_px_total"] >= REQUIRED_PX
        axp.set_title(f"z = {r['z_mm']:g} mm, {r['layers']} masks", fontsize=8.5,
                      color=INK, pad=4)
        axp.set_xlabel(f"{r['val_acc']:.4f}", fontsize=11, color=GOOD if ok else BAD,
                       labelpad=4, fontweight="bold")

    fig.text(0.055, 0.008,
             "Detector plane for one fixed digit, same digit in every column, intensity gamma 0.4. "
             "The last three columns carry identical reach and identical wrap error; only the split differs.\n"
             "Accuracies are the reduced ranking protocol (20k x 12 epochs) and are not comparable to the "
             "60k x 40 epoch deliverable number; the deepest configuration had not converged when it stopped.",
             fontsize=6.2, color=MUTED, va="bottom", linespacing=1.6)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def build_js(doc, out=OUT_JS):
    """Emit the widget bundle: measured points only, geometry is recomputed live."""
    z_arm, iso, floor = split_arms(doc)
    geom = {(g["z_mm"], g["layers"]): g for g in doc.get("geometry", [])}

    def point(r):
        g = geom.get((r["z_mm"], r["layers"]), {})
        return {"z_mm": r["z_mm"], "layers": r["layers"], "acc": r["val_acc"],
                "reach_total": r["reach_px_total"], "reach_hop": r["reach_px_per_hop"],
                "params": r["n_params"],
                "wrap_logit": g.get("logit_error"), "wrap_plane": g.get("plane_error")}

    bundle = {
        "grid": GRID, "dx": DX, "wavelength": WAVELENGTH,
        "input_lo": INPUT_LO, "input_hi": INPUT_HI, "required_px": REQUIRED_PX,
        "z_crit_mm": GRID * DX**2 / WAVELENGTH * 1e3,
        "floor_acc": floor["val_acc"] if floor else None,
        "protocol": z_arm[0]["protocol"] if z_arm else None,
        # The slider varies z at the shipped mask count, so its curve is the z arm
        # alone -- mixing in the iso points would stack four accuracies on one x.
        "points": [point(r) for r in z_arm],
        "iso_reach_px": ISO_TARGET_PX,
        "iso": [point(r) for r in iso],
        # Every 128-grid run in the sweep, deepest included, for the scaling curve
        # on /optics. `iso` above is only the four runs sitting exactly on the
        # iso-reach target, which is the right set for the reach argument and the
        # wrong one for "what does depth actually buy" -- that needs the 20, 28,
        # 40, 56 and 80-mask runs too. Sorted by depth, since the curve is read
        # left to right.
        # 256-grid runs carry an "n256_" config prefix; they belong to the
        # depth-versus-resolution question, not to this curve.
        "scaling": sorted(
            (point(r) for r in doc["runs"]
             if r["layers"] > 0 and not str(r["config"]).startswith("n256_")),
            key=lambda p: (p["layers"], p["z_mm"]),
        ),
        # The two full-budget runs, which are NOT part of the sweep: 60k images
        # rather than 20k, and the numbers every other page quotes. They are what
        # makes the scaling curve legible, because the sweep's own accuracies are
        # depressed by its short protocol and must never be compared to them.
        "full": [
            {"layers": 5, "acc": 0.7990, "label": "5 masks"},
            {"layers": 56, "acc": 0.9040, "label": "56 masks"},
        ],
    }

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        "/*\n"
        " * optics_sweep.js -- measured accuracies from the Phase-2 optics sweep.\n"
        " *\n"
        " * GENERATED by apps/sweep_report.py from exports/sweep/optics_sweep.json --\n"
        " * do not edit by hand. Committed on purpose: exports/ is gitignored, so this\n"
        " * is the repo's only record of what the sweep measured, and the only way\n"
        " * apps/web/optics.js has data in a fresh clone.\n"
        " *\n"
        " * Accuracies are the reduced ranking protocol (see .protocol), NOT the\n"
        " * 60k x 40 epoch deliverable number.\n"
        " */\n"
        f"var OPTICS_SWEEP = {json.dumps(bundle, indent=2)};\n"
        "if (typeof module !== 'undefined') { module.exports = OPTICS_SWEEP; }\n",
        encoding="utf-8")
    print(f"wrote {out}")


def main():
    doc = load_sweep()
    build_figure(doc)
    build_js(doc)


if __name__ == "__main__":
    main()
