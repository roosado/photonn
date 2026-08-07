"""Cross-check the candidate (14-mask) browser model against torch.

``apps/web/d2nn_sweep_weights.js`` ships the sweep's deeper network so it can run
beside the shipped one in ``apps/web/d2nn_compare.js``. Two things could silently
go wrong with it and neither would break any existing test: the **8-bit phase
decode** in ``d2nn.js:buildNet`` could disagree with the quantiser in
``apps/export_sweep_web.py``, and ``buildNet`` could mis-handle a bundle whose
geometry differs from the shipped one (14 masks at 2 mm, not 5 at 3 mm).

This rebuilds the torch model *from the committed bundle itself* -- decoding the
same uint8 codes the browser decodes -- and requires identical predictions. It
needs no gitignored file, so it runs on a fresh clone.

Requires Node on PATH; skips cleanly if absent.
"""
import base64
import json
import os
import re
import shutil
import subprocess

import numpy as np
import pytest
import torch

from photonn.models import D2NN
from photonn.train import encode_input

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "..", "apps", "web", "d2nn_sweep_weights.js")

LOGIT_TOL = 1e-3        # same bar as the shipped-model cross-check

node = shutil.which("node")
pytestmark = [
    pytest.mark.skipif(node is None, reason="node not on PATH; candidate cross-check skipped"),
    pytest.mark.skipif(not os.path.exists(BUNDLE),
                       reason="d2nn_sweep_weights.js not generated "
                              "(run apps.export_sweep_web)"),
]

_RUNNER = """
const NET = require('%s');
const deep = NET.buildNet(require('%s'));
const out = [];
for (let k = 0; k < deep.nGallery; k++) {
  const r = deep.classify(deep.galleryDigit(k));
  out.push({pred: r.pred, logits: Array.from(r.logits), label: deep.galleryLabel(k)});
}
process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def bundle():
    text = open(BUNDLE, encoding="utf-8").read()
    body = re.search(r"var W = (\{.*?\});\n", text, re.S)
    assert body, "d2nn_sweep_weights.js is not a generated bundle"
    return json.loads(body.group(1))


@pytest.fixture(scope="module")
def js(bundle):
    d2nn = os.path.join(HERE, "..", "apps", "web", "d2nn.js").replace("\\", "/")
    swj = BUNDLE.replace("\\", "/")
    proc = subprocess.run([node, "-e", _RUNNER % (d2nn, swj)], capture_output=True, text=True)
    assert proc.returncode == 0, f"node runner failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def _torch_logits(bundle):
    """Rebuild the model from the bundle's own uint8 codes, as the browser does."""
    n, layers = bundle["n"], bundle["n_layers"]
    codes = np.frombuffer(base64.b64decode(bundle["masks_b64"]), dtype=np.uint8)
    step = 2 * np.pi / 256
    masks = (codes.astype(np.float64) * step - np.pi).reshape(layers, n, n)

    model = D2NN(n=n, n_layers=layers, dx=bundle["dx"],
                 wavelength=bundle["wavelength"], separation=bundle["separation"])
    for m, phi in zip(model.masks, masks):
        m.phi.data = torch.as_tensor(phi, dtype=m.phi.dtype)

    digits = np.frombuffer(base64.b64decode(bundle["gallery_b64"]), dtype=np.uint8)
    digits = digits.reshape(-1, 28, 28).astype(np.float32) / 255.0
    with torch.no_grad():
        field = encode_input(torch.as_tensor(digits), scheme="both", n=n)
        return model(field).numpy()


def test_candidate_matches_torch(bundle, js):
    """Predictions identical and logits within 1e-3, through the 8-bit decode."""
    ref = _torch_logits(bundle)
    assert len(js) == len(ref)
    for k, row in enumerate(js):
        assert row["pred"] == int(ref[k].argmax()), f"digit {k}: JS {row['pred']} vs torch"
        err = float(np.abs(np.array(row["logits"]) - ref[k]).max())
        assert err < LOGIT_TOL, f"digit {k}: max logit error {err:.2e}"


def test_candidate_geometry_is_the_swept_one(bundle):
    """Guard the thing that makes this a *different* machine, not a re-export."""
    assert bundle["n_layers"] == 14
    assert bundle["separation"] == pytest.approx(2e-3)
    assert bundle["masks_bits"] == 8
    codes = base64.b64decode(bundle["masks_b64"])
    assert len(codes) == 14 * bundle["n"] ** 2, "mask payload is not one byte per phase"


def test_candidate_is_labelled_unshipped(bundle):
    """The number on this model must never be mistakable for a test accuracy.

    It is a short-protocol validation figure; the widget prints the provenance
    beside it, and this keeps the exporter from quietly dropping that.
    """
    prov = bundle.get("provenance", {})
    assert prov.get("shipped") is False
    assert "not_scored_on" in prov and "test set" in prov["not_scored_on"]
    assert prov["protocol"]["n_train"] < 60000, "provenance claims the full training set"


def test_candidate_shares_the_shipped_gallery(bundle):
    """Both models must be shown the same digits or the comparison is not one."""
    shipped_js = open(os.path.join(HERE, "..", "apps", "web", "d2nn_weights.js"),
                      encoding="utf-8").read()
    shipped = json.loads(re.search(r"var W = (\{.*?\});\n", shipped_js, re.S).group(1))
    assert bundle["gallery_b64"] == shipped["gallery_b64"]
    assert bundle["gallery_labels"] == shipped["gallery_labels"]
    assert bundle["regions"] == shipped["regions"]
