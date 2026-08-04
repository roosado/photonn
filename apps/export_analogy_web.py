"""Export the free-space/chip correspondence geometry for the browser.

The Phase-3 explainer widget (``apps/web/analogy.js``) argues that the diffractive
network and the MZI mesh are the same alternating machine,

    [ trainable phase ] -> [ fixed mixing ] -> ... -> |.|^2

and that they differ on exactly one axis: **how far one mixing layer reaches.**
That argument is quantitative, so every number it draws has to come from the
models rather than from the figure's author. This module reads them and writes

    apps/web/analogy_geom.js

- the D2NN half from the Phase-2 handoff (``exports/d2nn_phase2.h5``) and
  :func:`photonn.detect.default_regions`, with the per-hop reach from
  :func:`photonn.propagate.diffraction_reach_px`;
- the mesh half from ``layers.MZIMeshLayer._schedule`` itself, so the drawn
  columns *are* the trained topology and cannot drift from it.

**The generated file is committed**, for the same reason ``d2nn_weights.js`` is:
``*.h5``/``*.pt`` are gitignored, so a fresh clone must still be able to rebuild
the site. ``tests/test_correspondence.py`` re-derives every number and fails if
the committed file disagrees.

Run from the repo root in the project venv::

    python -m apps.export_analogy_web
"""
from __future__ import annotations

import json
import os

import h5py
import numpy as np

from photonn.detect import default_regions
from photonn.layers import MZIMeshLayer
from photonn.propagate import diffraction_reach_px

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D2NN_H5 = os.path.join(_REPO, "exports", "d2nn_phase2.h5")
MESH_H5 = os.path.join(_REPO, "exports", "mesh_phase3.h5")
OUT_JS = os.path.join(_REPO, "apps", "web", "analogy_geom.js")

SCHEMA = "analogy-geom/1"


def _acc_from_description(attrs) -> float:
    """Pull ``test_acc=...`` out of a handoff's free-text description attribute."""
    for token in str(attrs["description"]).split("|"):
        key, _, value = token.strip().partition("=")
        if key == "test_acc":
            return float(value)
    raise KeyError("no test_acc in the handoff description")


def d2nn_geometry() -> dict:
    """Reach and connectivity of the trained D2NN, derived from its handoff.

    Reads only the metadata groups -- the masks and the frozen test set are large
    and irrelevant here.
    """
    with h5py.File(D2NN_H5, "r") as f:
        geo, op = dict(f["geometry"].attrs), dict(f["operating_point"].attrs)
        seps = f["geometry"]["layer_separations_m"][...]
        accuracy = _acc_from_description(f.attrs)

    n, n_layers = int(geo["grid_size"]), int(geo["n_layers"])
    dx, lam = float(op["pixel_pitch_m"]), float(op["wavelength_m"])
    input_frac = float(op["input_frac"])
    if not np.allclose(seps, seps[0]):
        raise ValueError(f"non-uniform layer separations {seps}; the figure assumes one z.")
    z = float(seps[0])
    hops = n_layers + 1
    if len(seps) != hops:
        raise ValueError(f"{len(seps)} separations for {n_layers} masks; expected {hops}.")

    # Input aperture: the same central window photonn.train._embed writes into.
    win = max(1, int(round(input_frac * n)))
    off = (n - win) // 2
    input_window = [off, off + win - 1]                    # inclusive pixel indices

    # Detector patches: the fixed layout the model was trained and scored with.
    regions = [[r.y0, r.y1, r.x0, r.x1] for r in default_regions(n)]
    det_x = [min(r[2] for r in regions), max(r[3] - 1 for r in regions)]
    det_y = [min(r[0] for r in regions), max(r[1] - 1 for r in regions)]

    # Worst case, per axis: an input pixel at one edge of the window must be able
    # to reach the detector pixel farthest from it. Reach is per-axis (the FFT
    # band is a square in (fx, fy)), so x and y are separate one-dimensional tests.
    def required(window, extent):
        return max(window[1] - extent[0], extent[1] - window[0])

    need_x, need_y = required(input_window, det_x), required(input_window, det_y)
    need = max(need_x, need_y)

    reach_hop = diffraction_reach_px(n, dx, lam, z)
    reach_total = hops * reach_hop
    # Separation at which the stack stops being able to connect every input pixel
    # to every detector pixel. Below it, part of the input is physically invisible
    # to part of the readout, whatever the masks are set to.
    z_min = need * 2.0 * dx * dx / (hops * lam)

    return dict(
        n=n, dx=dx, wavelength=lam, separation=z, n_layers=n_layers, hops=hops,
        input_frac=input_frac, input_window=input_window,
        regions=regions, detector_x=det_x, detector_y=det_y,
        reach_px_per_hop=reach_hop, reach_px_total=reach_total,
        required_px_x=need_x, required_px_y=need_y, required_px=need,
        margin_px=reach_total - need,
        min_separation_m=z_min, z_crit_m=n * dx * dx / lam,
        accuracy=accuracy,
    )


def mesh_geometry(n_modes: int = 36) -> dict:
    """Clements topology of the trained mesh, read from the layer that defines it."""
    with h5py.File(MESH_H5, "r") as f:
        accuracy = _acc_from_description(f.attrs)
        modes = int(dict(f["operating_point"].attrs)["n_modes"])
        n_phases = int(f["parameters/phase_theta"].shape[0])
    if modes != n_modes:
        raise ValueError(f"handoff has {modes} modes, asked for {n_modes}.")

    layer = MZIMeshLayer(n_modes)
    columns = [[int(m) for m, _ in pairs] for pairs in layer._schedule]
    if sum(len(c) for c in columns) != layer.n_mzi:
        raise ValueError("schedule pair count disagrees with n_mzi.")
    # The handoff stores concat[V, U] phases, so the trained SVD network really
    # is two of these meshes -- check rather than assume.
    if n_phases != 2 * layer.n_mzi:
        raise ValueError(f"handoff has {n_phases} MZI phases, expected 2 x {layer.n_mzi}.")

    return dict(
        n_modes=n_modes,
        n_columns=len(columns),
        n_mzi=int(layer.n_mzi),
        columns=columns,
        reach_modes_per_column=1,          # a coupler touches its neighbour, no further
        required_columns=n_modes - 1,      # worst pair: mode 0 <-> mode n-1
        svd_columns=2 * len(columns),      # U . Sigma . V+ -> two meshes in series
        svd_mzi=2 * int(layer.n_mzi),
        accuracy=accuracy,
    )


_HEADER = """/*
 * analogy_geom.js -- geometry for the free-space/chip correspondence figure.
 *
 * GENERATED by apps/export_analogy_web.py -- do not edit by hand; re-run the
 * exporter instead. Committed because exports/*.h5 are gitignored, so this is
 * the only in-repo copy and the only way apps.build_site rebuilds the page from
 * a fresh clone. tests/test_correspondence.py re-derives every number here.
 *
 * d2nn : reach in *pixels per free-space hop*, the detector layout, and how far
 *        the design actually needs light to travel (all lengths in metres).
 * mesh : the Clements schedule itself -- columns[c] lists the top mode of every
 *        MZI in column c -- so the drawing is the trained topology.
 */
"""


def write_geom_js(path: str = OUT_JS) -> str:
    geom = {"schema": SCHEMA, "d2nn": d2nn_geometry(), "mesh": mesh_geometry()}
    body = json.dumps(geom, indent=2, sort_keys=True)
    text = (
        _HEADER
        + "(function () {\n  \"use strict\";\n  var G = "
        + body.replace("\n", "\n  ")
        + ";\n  if (typeof module !== \"undefined\" && module.exports) module.exports = G;\n"
        + "  if (typeof window !== \"undefined\") window.PHOTONN_ANALOGY_GEOM = G;\n})();\n"
    )
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


def main():
    d, m = d2nn_geometry(), mesh_geometry()
    print("D2NN")
    print(f"  {d['hops']} hops of {d['separation']*1e3:.1f} mm at "
          f"{d['wavelength']*1e9:.0f} nm, {d['n']}^2 grid, dx={d['dx']*1e6:.0f} um")
    print(f"  reach {d['reach_px_per_hop']:.3f} px/hop -> {d['reach_px_total']:.3f} px total")
    print(f"  input window px {d['input_window']}, detector pixels "
          f"x {d['detector_x']}, y {d['detector_y']}")
    print(f"  required {d['required_px_x']} px (x) / {d['required_px_y']} px (y) "
          f"-> margin {d['margin_px']:+.3f} px "
          f"({100*d['margin_px']/d['required_px']:+.2f}%)")
    print(f"  connectivity fails below separation {d['min_separation_m']*1e3:.4f} mm "
          f"(design {d['separation']*1e3:.3f} mm)")
    print("mesh")
    print(f"  {m['n_modes']} modes, {m['n_columns']} columns, {m['n_mzi']} MZIs; "
          f"SVD {m['svd_columns']} columns, {m['svd_mzi']} MZIs")
    print(f"  reach {m['reach_modes_per_column']} mode/column -> "
          f"{m['required_columns']} columns needed, {m['n_columns']} present")

    path = write_geom_js()
    print(f"\nwrote {path} ({os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    main()
