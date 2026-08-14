"""Export a sweep configuration's masks as a second browser bundle.

The optics sweep found that spending a fixed reach budget on more masks beats
spending it on distance (``docs/phase2_dnn.md``). This makes that comparison
*operable*: the deeper network runs in the browser next to the 5-mask one, on
the same digits, in ``apps/web/d2nn_compare.js``.

Two things this bundle is deliberately not:

* **Not a promotion.** The masks come from the ranking sweep (20k images, 12
  epochs) and have never been scored on the frozen test set. The bundle records
  that in ``provenance`` and the widget prints it, because a model a visitor can
  operate invites its number being read as a test accuracy.
* **Not float32.** Phases are quantised to 8 bits, which
  ``docs/tolerance_d2nn.md`` measures as free for this design (it holds to 3-bit
  phase control) and which is what a real SLM offers anyway. It is also what
  keeps a 14-mask model to ~300 KB instead of 1.2 MB.

The gallery is copied from ``apps/web/d2nn_weights.js`` rather than re-picked, so
both models are shown the *same* digits and the comparison is a comparison.

    python -m apps.export_sweep_web --config z2mm_L14
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from apps.web_bundle import encode_masks, read_bundle
from photonn.detect import default_regions
from photonn.propagate import diffraction_reach_px

_REPO = Path(__file__).resolve().parent.parent
SHIPPED_JS = _REPO / "apps" / "web" / "d2nn_weights.js"
SWEEP_DIR = _REPO / "exports" / "sweep"
OUT_JS = _REPO / "apps" / "web" / "d2nn_sweep_weights.js"

#: Grid of a sweep run that predates ``--grid`` (its records carry no "grid" key).
LEGACY_GRID, DX, WAVELENGTH = 128, 8e-6, 532e-9
PHASE_SCALE, INPUT_FRAC, READOUT_GAIN = float(np.pi), 0.5, 10.0

#: Region coordinates are integers, so the same layout at two grid sizes agrees
#: only up to rounding -- one pixel out of `n`, doubled because both edges round.
LAYOUT_TOL = 2.0


def parse_args():
    p = argparse.ArgumentParser(description="Export sweep masks for the browser.")
    p.add_argument("--config", default="z2mm_L14",
                   help="sweep config tag, e.g. z2mm_L14 (masks_<tag>.npy)")
    p.add_argument("--out", default=str(OUT_JS))
    return p.parse_args()


def check_layout(regions, grid, shipped) -> None:
    """Both models must read the *same detector layout*, or it is not a comparison.

    The original form of this check required the region pixel boxes to be
    identical to the shipped bundle's. That is right only while every model
    shares one grid. ``default_regions`` places patches as fractions of ``n``, so
    a 256 model has the same layout at twice the pixel coordinates -- identical
    boxes would mean the *layout* had shrunk by half.

    So the comparison is made in normalised coordinates, which is what "same
    layout" actually means, and still fails on a real layout change (a different
    ``field_frac`` or ``patch_frac``) at any grid.
    """
    n_shipped = shipped["n"]
    if len(regions) != len(shipped["regions"]):
        raise SystemExit(f"{len(regions)} detector regions against the shipped bundle's "
                         f"{len(shipped['regions'])}; not a like-for-like comparison.")

    worst, where = 0.0, None
    for c, (mine, theirs) in enumerate(zip(regions, shipped["regions"])):
        for a, b in zip(mine, theirs):
            # Compare as a fraction of the grid, then re-express in pixels of the
            # model being exported, so the tolerance means the same thing either way.
            delta = abs(a / grid - b / n_shipped) * grid
            if delta > worst:
                worst, where = delta, c
    if worst > LAYOUT_TOL:
        raise SystemExit(
            f"detector layout differs from the shipped bundle by {worst:.1f} px "
            f"(class {where}, grid {grid} vs {n_shipped}); the comparison would not be "
            "like for like. Re-export the shipped bundle first if the layout moved on purpose."
        )
    print(f"  layout matches the shipped bundle to {worst:.2f} px "
          f"(grid {grid} vs {n_shipped})")


def sweep_record(tag: str) -> dict:
    """The measured result for this config, straight from the sweep's own JSON."""
    doc = json.loads((SWEEP_DIR / "optics_sweep.json").read_text())
    for r in doc.get("runs", []):
        if r["config"] == tag:
            return r
    raise SystemExit(f"no run named {tag!r} in optics_sweep.json")


def main():
    args = parse_args()
    masks_path = SWEEP_DIR / f"masks_{args.config}.npy"
    if not masks_path.exists():
        raise SystemExit(f"no masks at {masks_path}; run apps.sweep_optics first.")

    masks = np.load(masks_path).astype(np.float64)
    run = sweep_record(args.config)
    n_layers, z = int(run["layers"]), run["z_mm"] * 1e-3
    # Runs written before --grid existed carry no "grid" key and were all 128.
    grid = int(run.get("grid", LEGACY_GRID))
    if masks.shape != (n_layers, grid, grid):
        raise SystemExit(f"masks {masks.shape} do not match n_layers={n_layers}, n={grid}.")

    masks_b64, max_err = encode_masks(masks, bits=8)
    shipped = read_bundle(SHIPPED_JS)
    regions = [[r.y0, r.y1, r.x0, r.x1] for r in default_regions(grid, 10)]
    check_layout(regions, grid, shipped)

    bundle = {
        "n": grid, "dx": DX, "wavelength": WAVELENGTH, "separation": z,
        "n_layers": n_layers, "readout_gain": READOUT_GAIN,
        "phase_scale": PHASE_SCALE, "input_frac": INPUT_FRAC,
        "regions": regions,
        "masks_bits": 8,
        "masks_b64": masks_b64,
        # Same digits as the shipped model, so switching compares machines only.
        "gallery_b64": shipped["gallery_b64"],
        "gallery_labels": shipped["gallery_labels"],
        "gallery_size": shipped["gallery_size"],
        # Same shape as the shipped bundle's block (apps/export_d2nn_web.py:
        # provenance), so d2nn_compare.js renders both columns' captions through
        # one code path and neither model's number lives in JavaScript.
        "provenance": {
            # Named for its depth, like every other column. What sets this bundle
            # apart is `not_scored_on`, which is what actually makes its number
            # incomparable -- not any judgement about its standing.
            "label": f"{n_layers} masks",
            "config": args.config,
            "accuracy": float(run["val_acc"]),
            "protocol": run["protocol"],
            "reach_px_total": run["reach_px_total"],
            "n_params": run["n_params"],
            "scored_on": "a held-out validation split carved from the MNIST train set",
            "not_scored_on": "the frozen 2000-image test set",
            "shipped": False,
            "caveat": "a ranking run, never scored on the test set, and not converged",
        },
    }

    reach = diffraction_reach_px(grid, DX, WAVELENGTH, z) * (n_layers + 1)
    header = f"""/*
 * d2nn_sweep_weights.js -- the {n_layers}-mask diffractive network, for the browser.
 *
 * GENERATED by apps/export_sweep_web.py from exports/sweep/masks_{args.config}.npy --
 * do not edit by hand. Committed because exports/ is gitignored.
 *
 * {n_layers} masks at {run['z_mm']:g} mm on a {grid}x{grid} grid ({grid * DX * 1e3:.3f} mm across),
 * {reach:.1f} px of total reach, {run['n_params']:,} phases quantised to 8 bits
 * (max phase error {max_err:.4f} rad; the Phase-4 budget holds this design to
 * 3-bit phase control, so 8 bits is free -- and is what an SLM offers).
 *
 * Validation accuracy {run['val_acc']:.4f} under the sweep's short ranking protocol
 * ({run['protocol']['n_train']} images, {run['protocol']['epochs']} epochs); it has never been scored on the frozen
 * test set, and it had not converged. Its number and the frozen-test-set numbers
 * quoted elsewhere are not the same measurement, which is what `not_scored_on`
 * records and what the widget's caption says.
 */
"""
    out = Path(args.out)
    out.write_text(
        header
        + "(function () {\n  \"use strict\";\n  var W = "
        + json.dumps(bundle, indent=2).replace("\n", "\n  ")
        + ";\n"
        "  if (typeof module !== \"undefined\" && module.exports) module.exports = W;\n"
        "  if (typeof window !== \"undefined\") window.D2NN_SWEEP_WEIGHTS = W;\n"
        "})();\n",
        encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    print(f"  {n_layers} masks, z={run['z_mm']:g} mm, val {run['val_acc']:.4f}, "
          f"max quantisation error {max_err:.4f} rad")


if __name__ == "__main__":
    main()
