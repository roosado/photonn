"""Export a sweep configuration's masks as a second browser bundle.

The optics sweep found that spending a fixed reach budget on more masks beats
spending it on distance (``docs/phase2_dnn.md``). This makes that comparison
*operable*: the deeper network runs in the browser next to the shipped one, on
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
import base64
import json
import re
from pathlib import Path

import numpy as np

from photonn.detect import default_regions
from photonn.propagate import diffraction_reach_px

_REPO = Path(__file__).resolve().parent.parent
SHIPPED_JS = _REPO / "apps" / "web" / "d2nn_weights.js"
SWEEP_DIR = _REPO / "exports" / "sweep"
OUT_JS = _REPO / "apps" / "web" / "d2nn_sweep_weights.js"

GRID, DX, WAVELENGTH = 128, 8e-6, 532e-9
PHASE_SCALE, INPUT_FRAC, READOUT_GAIN = float(np.pi), 0.5, 10.0


def parse_args():
    p = argparse.ArgumentParser(description="Export sweep masks for the browser.")
    p.add_argument("--config", default="z2mm_L14",
                   help="sweep config tag, e.g. z2mm_L14 (masks_<tag>.npy)")
    p.add_argument("--out", default=str(OUT_JS))
    return p.parse_args()


def shipped_bundle(path=SHIPPED_JS) -> dict:
    """Parse the committed bundle -- same trick as apps/analogy_figure.load_geom."""
    text = Path(path).read_text(encoding="utf-8")
    body = re.search(r"var W = (\{.*?\});\n", text, re.S)
    if body is None:
        raise SystemExit(f"{path} does not look like a generated d2nn_weights bundle.")
    return json.loads(body.group(1))


def quantise_phase(masks: np.ndarray) -> tuple[str, float]:
    """Wrap to [-pi, pi) and encode as uint8 codes; return (base64, max error)."""
    wrapped = (masks + np.pi) % (2 * np.pi) - np.pi
    step = 2 * np.pi / 256
    codes = np.rint((wrapped + np.pi) / step).astype(np.int64)
    codes = np.clip(codes, 0, 255).astype(np.uint8)
    recovered = codes.astype(np.float64) * step - np.pi
    return base64.b64encode(codes.tobytes()).decode("ascii"), float(np.abs(recovered - wrapped).max())


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
    if masks.shape != (n_layers, GRID, GRID):
        raise SystemExit(f"masks {masks.shape} do not match n_layers={n_layers}, n={GRID}.")

    masks_b64, max_err = quantise_phase(masks)
    shipped = shipped_bundle()
    regions = [[r.y0, r.y1, r.x0, r.x1] for r in default_regions(GRID, 10)]
    if regions != shipped["regions"]:
        raise SystemExit("detector regions differ from the shipped bundle; the comparison "
                         "would not be like for like.")

    bundle = {
        "n": GRID, "dx": DX, "wavelength": WAVELENGTH, "separation": z,
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
            "label": "Candidate",
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

    reach = diffraction_reach_px(GRID, DX, WAVELENGTH, z) * (n_layers + 1)
    header = f"""/*
 * d2nn_sweep_weights.js -- a *candidate* diffractive network, for the browser.
 *
 * GENERATED by apps/export_sweep_web.py from exports/sweep/masks_{args.config}.npy --
 * do not edit by hand. Committed because exports/ is gitignored.
 *
 * {n_layers} masks at {run['z_mm']:g} mm ({reach:.1f} px of total reach, matching the shipped
 * model's best-performing separation budget), {run['n_params']:,} phases quantised to
 * 8 bits (max phase error {max_err:.4f} rad; the Phase-4 budget holds this design
 * to 3-bit phase control, so 8 bits is free -- and is what an SLM offers).
 *
 * NOT SHIPPED. Validation accuracy {run['val_acc']:.4f} under the sweep's short ranking
 * protocol ({run['protocol']['n_train']} images, {run['protocol']['epochs']} epochs); it has never been scored on
 * the frozen test set, and it had not converged. It is here to be compared
 * against, not quoted.
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
