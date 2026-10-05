"""The activation page's live chip must be the chip that was trained.

``apps/web/deep_mesh_weights.js`` carries the Phase-5 two-layer mesh at 16 bits, and
``activation.js`` runs it through ``mesh.js`` one gallery digit at a time. Three ways
for that to go quietly wrong, each guarded here:

* **the bundle drifts from the model** -- it is committed because ``exports/`` is
  not, so nothing forces a regeneration when the model changes;
* **the JavaScript rebuild is wrong** -- a transposed index, the V and U halves of
  a layer swapped, the output screens read from the wrong layer, the activation's
  sign convention flipped. Any of those draws a confident picture of another chip;
* **the power is wrong** -- the activation responds to watts, and a forgotten
  ``sqrt(P)`` puts it at one watt. In the cubic tail the *logits* are
  scale-covariant and may not move at all when that happens, so the check is on
  the powers entering the activation: ten times the input power must give exactly
  ten times those.

The reference is ``photonn.mzi.deep_mesh_forward`` on the decoded codes, which is
what MATLAB reproduces to 1e-13 and PyTorch to float32.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

import numpy as np
import pytest

from apps.export_deep_mesh_web import LABEL, TWO_PI, decode
from apps.web_bundle import read_bundle
from photonn import mzi

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RUNNER = os.path.join(HERE, "deep_mesh_runner.js")
WEIGHTS_JS = os.path.join(REPO, "apps", "web", "deep_mesh_weights.js")
WIDGET_JS = os.path.join(REPO, "apps", "web", "activation.js")
FIXTURE = os.path.join(HERE, "fixtures", "deep_mesh_reference.json")
HANDOFF = os.path.join(REPO, "exports", "deep_mesh_phase5.h5")

#: Logit agreement. JS and NumPy both run float64 on the same decoded codes; the
#: inputs are float32 in the bundle and the reference reads the same float32s.
#: Measured ~1e-13; this leaves room without hiding a real error.
LOGIT_TOL = 1e-9

node = shutil.which("node")


@pytest.fixture(scope="module")
def bundle():
    return read_bundle(WEIGHTS_JS)


@pytest.fixture(scope="module")
def reference():
    return json.load(open(FIXTURE, encoding="utf-8"))


@pytest.fixture(scope="module")
def js():
    if node is None:
        pytest.skip("node not on PATH")
    proc = subprocess.run([node, RUNNER], capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, f"deep mesh runner failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def _params(bundle):
    L, n, nm = bundle["n_layers"], bundle["n"], bundle["n_mzi"]
    return {"phase_theta": decode(bundle["theta_b64"], TWO_PI, (L, 2 * nm)),
            "phase_phi": decode(bundle["phi_b64"], TWO_PI, (L, 2 * nm)),
            "sigma": decode(bundle["sigma_b64"], 1.0, (L, n)),
            "out_phase": decode(bundle["out_phase_b64"], TWO_PI, (L, 2, n))}


def _inputs(js, bundle):
    raw = np.asarray(js["inputs"], dtype=np.float64).reshape(-1, bundle["n"], 2)
    return raw[..., 0] + 1j * raw[..., 1]


def _forward(bundle, x, p_in):
    op = bundle["operating_point"]
    return mzi.deep_mesh_forward(_params(bundle), x, input_power_w=p_in,
                                 alpha=op["eo_alpha"], g_phi=op["eo_g_phi"],
                                 phi_b=op["eo_phi_b"])


def test_the_bundle_has_the_shape_of_a_two_layer_chip(bundle):
    assert bundle["n_layers"] == 2 and bundle["n"] == 16 and bundle["n_mzi"] == 120
    sigma = decode(bundle["sigma_b64"], 1.0, (2, 16))
    assert sigma.min() >= 0.0 and sigma.max() <= 1.0, "a passive chip cannot amplify"
    assert len(bundle["gallery_labels"]) == 16


def test_js_decodes_what_python_decodes(js, bundle):
    """Before comparing logits, so a mismatch there is not an encoding mismatch."""
    p = _params(bundle)
    for key, js_key in (("phase_theta", "theta"), ("phase_phi", "phi"), ("sigma", "sigma"),
                        ("out_phase", "outPhase")):
        assert np.abs(np.asarray(js["decoded"][js_key]) - p[key].reshape(-1)).max() < 1e-12


def test_every_gallery_digit_is_classified_as_the_reference_classifies_it(js, bundle, reference):
    x = _inputs(js, bundle)
    out, _ = _forward(bundle, x, bundle["operating_point"]["input_power_w"])
    inten = np.abs(out) ** 2
    ref = inten[:, :10] / inten.sum(axis=1, keepdims=True) * bundle["operating_point"]["readout_gain"]
    got = np.array([d["logits"] for d in js["digits"]])
    assert [d["pred"] for d in js["digits"]] == list(inten[:, :10].argmax(axis=1))
    assert [d["pred"] for d in js["digits"]] == reference["predictions"]
    assert np.abs(got - ref).max() < LOGIT_TOL
    # And the exporter's fixture, computed from float64 inputs before the float32 cast.
    assert np.abs(got - np.array(reference["logits"])).max() < 1e-4


def test_the_activation_sees_watts(js, bundle, reference):
    """Ten times the light in, exactly ten times the light at every activation."""
    x = _inputs(js, bundle)
    _, taps = _forward(bundle, x, bundle["operating_point"]["input_power_w"])
    got = np.array([d["taps"] for d in js["digits"]])
    got10 = np.array([d["taps10"] for d in js["digits"]])
    assert np.allclose(got, taps[0], rtol=1e-12, atol=0)
    assert np.allclose(got10, 10 * got, rtol=1e-12, atol=0)
    assert np.allclose(got, reference["activation_input_power_w"], rtol=1e-5, atol=0)
    # Watts, not a unit-norm field: at 1 mW in, every mode sees well under a milliwatt.
    assert got.max() < bundle["operating_point"]["input_power_w"]


def test_the_provenance_names_the_model_by_its_depth(bundle):
    prov = bundle["provenance"]
    assert prov["label"] == LABEL
    for word in ("shipped", "unshipped", "promoted", "candidate", "new"):
        assert word not in prov["label"].lower()
    assert set(prov["protocol"]) == {"n_train", "epochs", "seed"}
    assert "frozen" in prov["scored_on"]


def test_the_widget_quotes_no_accuracy_of_its_own(bundle):
    """Captions come from provenance; the number must not be baked into the JS."""
    src = open(WIDGET_JS, encoding="utf-8").read()
    acc = bundle["provenance"]["accuracy"]
    for places in (2, 3, 4):
        assert f"{acc:.{places}f}" not in src


@pytest.mark.skipif(not os.path.exists(HANDOFF), reason="exports/deep_mesh_phase5.h5 not present")
def test_the_committed_bundle_is_the_handoff_to_one_code(bundle):
    from photonn.handoff import read_parameters

    p = read_parameters(HANDOFF)
    step = TWO_PI / 65536
    for key, b64k in (("phase_theta", "theta_b64"), ("phase_phi", "phi_b64"),
                      ("out_phase", "out_phase_b64")):
        stored = np.mod(np.asarray(p[key], dtype=float), TWO_PI)
        dec = decode(bundle[b64k], TWO_PI, stored.shape)
        err = np.abs(np.angle(np.exp(1j * (dec - stored))))
        assert err.max() <= step, key
    sig = decode(bundle["sigma_b64"], 1.0, np.asarray(p["sigma"]).shape)
    assert np.abs(sig - p["sigma"]).max() <= 1 / 65536
