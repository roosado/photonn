"""Export the Phase-5 activated mesh for the browser.

The activation page (``site/activation.html``) carries the trained two-layer chip
live: pick a test digit and see where each of its sixteen modes lands on the
activation's curve, and what the ten detectors read. For that to be this chip and
not a cartoon of one, it has to run the trained settings, so this module reads them
out of the Phase-5 handoff and writes

    apps/web/deep_mesh_weights.js            (window.PHOTONN_DEEP_MESH)
    tests/fixtures/deep_mesh_reference.json  (what the browser must reproduce)

**Both are committed**, for the reason ``mesh_weights.js`` is: ``exports/`` is
gitignored, so these are the repo's only record of what the trained chip says and
the only way the page rebuilds from a fresh clone.

* **16-bit codes**, as for the 36-mode mesh. The page beside the widget quotes a
  0.03 rad phase edge, and an 8-bit code over 2*pi is a 0.025 rad step: the
  quantisation would be the size of the result under discussion.
* **The handoff is read only through** :mod:`photonn.handoff`, never ``h5py``.
* **The accuracy in the bundle is the 16-bit model's**, scored by the float64
  reference (:func:`photonn.mzi.deep_mesh_forward`) on the decoded codes over the
  whole frozen test set -- the model as shipped, not the float one it came from.
* **The reference fixture comes from the exported codes**, not from the ``.pt``
  checkpoint: the browser runs the codes, so the codes are what it is held to.

Run from the repo root in the project venv::

    python -m apps.export_deep_mesh_web
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import numpy as np

from apps.export_d2nn_web import pick_gallery
from apps.web_bundle import b64, write_bundle
from photonn import mzi
from photonn.handoff import read_handoff, read_parameters, read_test_inputs, read_test_set

_REPO = Path(__file__).resolve().parent.parent
DEEP_H5 = _REPO / "exports" / "deep_mesh_phase5.h5"
DEEP_PT = _REPO / "exports" / "deep_mesh_phase5.pt"
OUT_JS = _REPO / "apps" / "web" / "deep_mesh_weights.js"
FIXTURE = _REPO / "tests" / "fixtures" / "deep_mesh_reference.json"

SCHEMA = "deep-mesh-weights/1"
BITS = 16
LEVELS = 1 << BITS
TWO_PI = 2.0 * np.pi
#: The label a model is shown under: named for its depth, never its status.
LABEL = "2 layers + activation"


def _encode(values, span):
    """Quantise to little-endian 16-bit codes over ``[0, span)``, phases wrapped."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    codes = np.floor(np.mod(v, span) / span * LEVELS).astype(np.int64)
    return base64.b64encode(np.clip(codes, 0, LEVELS - 1).astype("<u2").tobytes()).decode("ascii")


def decode(b64s, span, shape):
    """Inverse of :func:`_encode`: what the browser's ``mesh.decode16`` computes."""
    codes = np.frombuffer(base64.b64decode(b64s), dtype="<u2").astype(np.float64)
    return ((codes + 0.5) / LEVELS * span).reshape(shape)


def _protocol():
    """n_train / epochs / seed, read off the checkpoint rather than restated."""
    import torch

    if not DEEP_PT.exists():
        raise SystemExit(f"{DEEP_PT} not found; the provenance protocol is read from it.")
    args = torch.load(DEEP_PT, weights_only=False)["args"]
    n_train = args["subset_train"] or 60000
    return {"n_train": int(n_train), "epochs": int(args["epochs"]), "seed": int(args["seed"])}


def build(path=DEEP_H5):
    """Read, check, quantise and score. Returns ``(payload, reference)``."""
    h = read_handoff(path)
    if h.model_type != "deep_mesh" or h.schema_version != "0.4.0":
        raise SystemExit(f"{path} is {h.model_type!r} at {h.schema_version}; need deep_mesh 0.4.0.")
    p = read_parameters(path)
    if str(p["mesh_order"]) != "V,U":
        raise SystemExit(f"mesh_order is {p['mesh_order']!r}; the widget composes U.diag(sigma).V")
    if str(p["topology"]) != "clements_rectangular":
        raise SystemExit(f"topology is {p['topology']!r}; the widget builds a Clements brick")
    sigma = np.asarray(p["sigma"], dtype=float)
    if sigma.min() < 0.0 or sigma.max() > 1.0:
        raise SystemExit(f"sigma runs {sigma.min():.4f}..{sigma.max():.4f}; a passive chip "
                         "cannot amplify. Re-export the handoff.")
    n_layers, n = sigma.shape
    n_mzi = n * (n - 1) // 2
    op = {k: h.op(k) for k in ("input_power_w", "eo_alpha", "eo_g_phi", "eo_phi_b",
                                "integration_time_s", "wavelength_m", "readout_gain")}

    enc = {"theta_b64": _encode(p["phase_theta"], TWO_PI),
           "phi_b64": _encode(p["phase_phi"], TWO_PI),
           "sigma_b64": _encode(np.clip(sigma, 0.0, 1.0 - 1e-12), 1.0),
           "out_phase_b64": _encode(p["out_phase"], TWO_PI)}
    shipped = {"phase_theta": decode(enc["theta_b64"], TWO_PI, (n_layers, 2 * n_mzi)),
               "phase_phi": decode(enc["phi_b64"], TWO_PI, (n_layers, 2 * n_mzi)),
               "sigma": decode(enc["sigma_b64"], 1.0, (n_layers, n)),
               "out_phase": decode(enc["out_phase_b64"], TWO_PI, (n_layers, 2, n))}

    x = read_test_inputs(path)
    images, labels = read_test_set(path)
    labels = np.asarray(labels).astype(int)

    def forward(params, xs):
        return mzi.deep_mesh_forward(params, xs, input_power_w=op["input_power_w"],
                                     alpha=op["eo_alpha"], g_phi=op["eo_g_phi"],
                                     phi_b=op["eo_phi_b"])

    out, _ = forward(shipped, x)
    inten = np.abs(out) ** 2
    preds = inten[:, :10].argmax(axis=1)
    acc16 = float(np.mean(preds == labels))
    out_f, _ = forward(p, x)
    acc_float = float(np.mean((np.abs(out_f) ** 2)[:, :10].argmax(axis=1) == labels))

    idx = pick_gallery(labels, preds)
    gx = x[idx]
    g_out, g_taps = forward(shipped, gx)
    g_int = np.abs(g_out) ** 2
    logits = g_int[:, :10] / g_int.sum(axis=1, keepdims=True) * op["readout_gain"]

    proto = _protocol()
    payload = {
        "schema": SCHEMA,
        "bits": BITS,
        "n": n,
        "n_layers": n_layers,
        "n_mzi": n_mzi,
        **enc,
        "operating_point": op,
        "gallery_b64": b64(np.round(np.asarray(images)[idx] * 255.0).astype(np.uint8), "u1"),
        "gallery_labels": [int(labels[i]) for i in idx],
        "gallery_size": 28,
        # The encoded unit-norm input per gallery digit, re/im interleaved. Shipped
        # rather than recomputed: the encoder is a 2-D Fourier transform the page
        # has no other reason to carry.
        "gallery_inputs_b64": b64(np.stack([gx.real, gx.imag], axis=-1).reshape(len(idx), -1), "<f4"),
        "provenance": {
            "label": LABEL,
            "accuracy": round(acc16, 4),
            "scored_on": f"the frozen {len(labels):,}-image MNIST test set, 16 Fourier modes",
            "protocol": proto,
        },
    }
    reference = {
        "gallery_index": [int(i) for i in idx],
        "labels": [int(labels[i]) for i in idx],
        "logits": logits.tolist(),
        "predictions": g_int[:, :10].argmax(axis=1).tolist(),
        "activation_input_power_w": g_taps[0].tolist(),
        "input_power_w": op["input_power_w"],
        "accuracy_16bit": acc16,
        "accuracy_float": acc_float,
        # Inputs as float32, exactly as the bundle carries them, so the test can
        # rerun the reference on what the browser actually reads.
        "note": "logits from mzi.deep_mesh_forward on the decoded 16-bit codes",
    }
    return payload, reference


_HEADER = """/*
 * deep_mesh_weights.js -- the trained Phase-5 chip, for the browser.
 *
 * GENERATED by apps/export_deep_mesh_web.py -- do not edit by hand; re-run the
 * exporter. Committed because exports/ is gitignored, so this is the only in-repo
 * copy of what the trained two-layer chip says. tests/test_deep_mesh_web.py runs it
 * under Node and holds it to tests/fixtures/deep_mesh_reference.json.
 *
 * Two SVD layers of 16 modes (120 MZIs per mesh, V then U per layer), with a bank of
 * Williamson et al. (2020) electro-optic activations between them. All parameters
 * are 16-bit little-endian codes, decoded as (code + 0.5) / 65536 * span:
 *
 *   theta_b64, phi_b64   [n_layers, 2*120] phases over [0, 2*pi)
 *   sigma_b64            [n_layers, 16] transmissions over [0, 1]
 *   out_phase_b64        [n_layers, 2, 16] output-screen phases over [0, 2*pi)
 *
 * operating_point is the handoff's: watts, rad/W, rad. The gallery ships each
 * digit's encoded input (16 complex Fourier coefficients, unit norm, float32 re/im).
 */
"""


def main():
    payload, reference = build()
    prov = payload["provenance"]
    print(f"{payload['n_layers']} layers x {payload['n']} modes; 16-bit accuracy "
          f"{prov['accuracy']:.4f} (float {reference['accuracy_float']:.4f}) on {prov['scored_on']}")
    write_bundle(OUT_JS, payload, header=_HEADER, window_name="PHOTONN_DEEP_MESH",
                 var_name="W", sort_keys=True)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(reference, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {OUT_JS} ({os.path.getsize(OUT_JS) / 1024:.1f} KB) and {FIXTURE}")


if __name__ == "__main__":
    main()
