"""Export the trained mesh's phase settings for the browser.

The `/chip` page carries one error-mechanism widget: coupler imbalance, the
fabrication error that has no meaning at all for a phase mask and that
``docs/tolerance_mesh.md`` measures binding level with phase error. The widget
shows what an imbalanced coupler does to the operator the chip computes,

    O = M_u . diag(sigma) . M_v

drawn as ``|O|`` before and after. For that to be a statement about *this* chip
rather than about a cartoon, it has to be driven by the trained settings, so this
module reads them out of the Phase-3 handoff and writes

    apps/web/mesh_weights.js

**The generated file is committed**, for the same reason ``d2nn_weights.js`` and
``analogy_geom.js`` are: ``exports/*.h5`` is gitignored, so this is the only
in-repo copy and the only way ``apps.build_site`` rebuilds `/chip` from a fresh
clone. ``tests/test_mesh_web.py`` decodes it and re-derives the operator.

Codes are **16-bit**, not the 8 bits the mask bundles use, and the reason is the
result on the page next to it. A phase mask is displayed as a colour wheel where
8 bits is finer than the eye resolves; here the same page states that the mesh
holds at 0.03 rad of phase error and fails at 0.05. An 8-bit code over 2*pi is a
0.025 rad step -- the same order as the tolerance being reported -- so the
picture would carry a quantisation error indistinguishable from the effect under
discussion. 16 bits is a 9.6e-5 rad step and costs 3.4 KB.

Run from the repo root in the project venv::

    python -m apps.export_mesh_web
"""
from __future__ import annotations

import base64
import json
import os

import h5py
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_H5 = os.path.join(_REPO, "exports", "mesh_phase3.h5")
OUT_JS = os.path.join(_REPO, "apps", "web", "mesh_weights.js")

SCHEMA = "mesh-weights/1"
BITS = 16
#: Codes are unsigned little-endian, decoded in JS as ``code / LEVELS * span``.
LEVELS = 1 << BITS
TWO_PI = 2.0 * np.pi


def _encode(values: np.ndarray, span: float) -> str:
    """Quantise ``values`` in ``[0, span)`` to little-endian 16-bit codes.

    Phases are wrapped rather than clipped: the physics is 2*pi-periodic, so a
    setting of 7.4 rad and one of 1.1 rad are the same coupler, and wrapping keeps
    every code inside the range instead of piling three of them onto the ceiling.
    """
    v = np.asarray(values, dtype=np.float64)
    codes = np.floor(np.mod(v, span) / span * LEVELS).astype(np.int64)
    codes = np.clip(codes, 0, LEVELS - 1).astype("<u2")
    return base64.b64encode(codes.tobytes()).decode("ascii")


def decode(b64: str, span: float) -> np.ndarray:
    """Inverse of :func:`_encode`, for the tests and for anyone reading this."""
    codes = np.frombuffer(base64.b64decode(b64), dtype="<u2").astype(np.float64)
    return (codes + 0.5) / LEVELS * span


def _acc_from_description(attrs) -> float:
    """Pull ``test_acc=...`` out of the handoff's free-text description."""
    for token in str(attrs["description"]).split("|"):
        key, _, value = token.strip().partition("=")
        if key == "test_acc":
            return float(value)
    raise KeyError("no test_acc in the handoff description")


def mesh_weights(path: str = MESH_H5) -> dict:
    """Read the trained settings out of the schema-0.2.0 mesh handoff."""
    with h5py.File(path, "r") as f:
        if str(f.attrs["schema_version"]) != "0.2.0":
            raise ValueError(
                f"{path} is schema {f.attrs['schema_version']}; the widget needs "
                "0.2.0, which is the first version to carry sigma and out_phase. "
                "Re-export with `python -m apps.train_mesh --export-only`."
            )
        p = f["parameters"]
        n_modes = int(p.attrs["n_modes"])
        n_mzi = int(p.attrs["n_mzi_per_mesh"])
        order = str(p.attrs["mesh_order"])
        topology = str(p.attrs["topology"])
        theta = p["phase_theta"][...]
        phi = p["phase_phi"][...]
        sigma = p["sigma"][...]
        out_phase = p["out_phase"][...]
        accuracy = _acc_from_description(f.attrs)

    if order != "V,U":
        raise ValueError(f"mesh_order is {order!r}; the widget composes U.Sigma.V")
    if topology != "clements_rectangular":
        raise ValueError(f"topology is {topology!r}; the widget builds a Clements brick")
    if theta.shape != (2 * n_mzi,) or phi.shape != theta.shape:
        raise ValueError(f"expected 2 x {n_mzi} phases, got {theta.shape} / {phi.shape}")
    if out_phase.shape != (2, n_modes):
        raise ValueError(f"out_phase is {out_phase.shape}, expected (2, {n_modes})")
    # Passivity is what makes |O| a picture of a buildable chip rather than of a
    # model with free gain in it. photonn.mzi.passivize guarantees it; assert it
    # here so a handoff exported before that lands cannot reach the page.
    if sigma.min() < 0.0 or sigma.max() > 1.0 + 1e-12:
        raise ValueError(
            f"sigma runs {sigma.min():.4f}..{sigma.max():.4f}; a passive mesh "
            "cannot amplify. Re-export -- photonn.mzi.passivize folds the signs "
            "into out_phase and divides the gain out."
        )

    return {
        "schema": SCHEMA,
        "bits": BITS,
        "n": n_modes,
        "n_mzi": n_mzi,
        "accuracy": accuracy,
        # Both meshes, V first, exactly as the handoff concatenates them.
        "theta_b64": _encode(theta, TWO_PI),
        "phi_b64": _encode(phi, TWO_PI),
        # sigma is already in [0, 1] after passivization, so its span is 1.
        "sigma_b64": _encode(np.clip(sigma, 0.0, 1.0 - 1e-12), 1.0),
        "out_phase_b64": _encode(out_phase.reshape(-1), TWO_PI),
    }


_HEADER = """/*
 * mesh_weights.js -- the trained MZI mesh's phase settings, for the browser.
 *
 * GENERATED by apps/export_mesh_web.py -- do not edit by hand; re-run the
 * exporter instead. Committed because exports/*.h5 are gitignored, so this is
 * the only in-repo copy and the only way apps.build_site rebuilds /chip from a
 * fresh clone. tests/test_mesh_web.py decodes it and rebuilds the operator.
 *
 * 630 MZIs per mesh, two meshes (V then U) around a diagonal sigma, all four
 * arrays quantised to 16-bit little-endian codes:
 *
 *   theta_b64, phi_b64  2*630 phases each, over [0, 2*pi)
 *   sigma_b64             36 transmissions, over [0, 1]
 *   out_phase_b64       2*36 output-screen phases, over [0, 2*pi), V then U
 *
 * Decode as  (code + 0.5) / 65536 * span.
 */
"""


def write_weights_js(path: str = OUT_JS) -> str:
    body = json.dumps(mesh_weights(), indent=2, sort_keys=True)
    text = (
        _HEADER
        + "(function () {\n  \"use strict\";\n  var W = "
        + body.replace("\n", "\n  ")
        + ";\n  if (typeof module !== \"undefined\" && module.exports) module.exports = W;\n"
        + "  if (typeof window !== \"undefined\") window.PHOTONN_MESH = W;\n})();\n"
    )
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


def main():
    w = mesh_weights()
    print(f"{w['n']} modes, {w['n_mzi']} MZIs per mesh, two meshes, "
          f"accuracy {w['accuracy']:.4f}")
    theta = decode(w["theta_b64"], TWO_PI)
    print(f"  theta {theta.min():.4f}..{theta.max():.4f} rad at "
          f"{TWO_PI / LEVELS:.2e} rad per code")
    path = write_weights_js()
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
