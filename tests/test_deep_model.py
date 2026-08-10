"""Cross-check the deep (56-mask) browser model against torch.

``apps/web/d2nn_deep_weights.js`` ships the optics sweep's deliverable retrain so
it can run beside the shipped model in ``apps/web/d2nn_compare.js``. It is the
first bundle that is both **quantised** and **deep**, and either property could
break quietly: the 8-bit decode in ``d2nn.js:buildNet`` could disagree with
``apps.web_bundle.encode_masks``, and 57 hops give a rounding error 11 times as
long to accumulate as the shipped 5-mask model's 6.

This mirrors ``tests/test_sweep_model.py``: rebuild the torch model *from the
committed bundle itself*, decoding the same uint8 codes the browser decodes, and
require identical predictions. It needs no gitignored file, so it runs on a
fresh clone.

The provenance assertions differ from the sweep candidate's on purpose. That one
was ranked on a validation split and must say it was never tested; this one *was*
scored on the frozen test set, at the full training budget, so its number and the
shipped 0.7990 are the same measurement. What it must not claim is that it is
shipped -- the Phase-4 budget prices its extra accuracy in fabrication tolerance
(``docs/tolerance_d2nn.md``), and promotion is a separate decision.

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

from apps.web_bundle import decode_masks
from photonn.models import D2NN
from photonn.train import encode_input

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "apps", "web")
BUNDLE = os.path.join(WEB, "d2nn_deep_weights.js")

LOGIT_TOL = 1e-3        # same bar as the shipped-model cross-check
N_LAYERS = 56           # the configuration this bundle exists to carry

node = shutil.which("node")
pytestmark = [
    pytest.mark.skipif(node is None, reason="node not on PATH; deep cross-check skipped"),
    pytest.mark.skipif(not os.path.exists(BUNDLE),
                       reason="d2nn_deep_weights.js not generated "
                              "(run apps.export_d2nn_web --out apps/web/d2nn_deep_weights.js)"),
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


def read_bundle(name):
    text = open(os.path.join(WEB, name), encoding="utf-8").read()
    body = re.search(r"var W = (\{.*?\});\n", text, re.S)
    assert body, f"{name} is not a generated bundle"
    return json.loads(body.group(1))


@pytest.fixture(scope="module")
def bundle():
    return read_bundle("d2nn_deep_weights.js")


@pytest.fixture(scope="module")
def js(bundle):
    d2nn = os.path.join(WEB, "d2nn.js").replace("\\", "/")
    proc = subprocess.run([node, "-e", _RUNNER % (d2nn, BUNDLE.replace("\\", "/"))],
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"node runner failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def _torch_logits(bundle):
    """Rebuild the model from the bundle's own phase codes, as the browser does."""
    n, layers = bundle["n"], bundle["n_layers"]
    masks = decode_masks(bundle["masks_b64"], bundle.get("masks_bits", 32), (layers, n, n))

    model = D2NN(n=n, n_layers=layers, dx=bundle["dx"],
                 wavelength=bundle["wavelength"], separation=bundle["separation"])
    for m, phi in zip(model.masks, masks):
        m.phi.data = torch.as_tensor(phi, dtype=m.phi.dtype)

    digits = np.frombuffer(base64.b64decode(bundle["gallery_b64"]), dtype=np.uint8)
    digits = digits.reshape(-1, 28, 28).astype(np.float32) / 255.0
    with torch.no_grad():
        field = encode_input(torch.as_tensor(digits), scheme="both", n=n)
        return model(field).numpy()


def test_deep_matches_torch(bundle, js):
    """Predictions identical and logits within 1e-3, through 57 hops of quantised phase."""
    ref = _torch_logits(bundle)
    assert len(js) == len(ref)
    for k, row in enumerate(js):
        assert row["pred"] == int(ref[k].argmax()), f"digit {k}: JS {row['pred']} vs torch"
        err = float(np.abs(np.array(row["logits"]) - ref[k]).max())
        assert err < LOGIT_TOL, f"digit {k}: max logit error {err:.2e}"


def test_deep_geometry_is_the_trained_one(bundle):
    """Guard what makes this a different machine: depth, and a sub-millimetre gap."""
    assert bundle["n_layers"] == N_LAYERS
    assert bundle["separation"] == pytest.approx(0.5263e-3, rel=1e-4)
    bits = bundle["masks_bits"]
    assert bits in (4, 8), f"a deep bundle must ship quantised, got masks_bits={bits}"
    payload = base64.b64decode(bundle["masks_b64"])
    expected = N_LAYERS * bundle["n"] ** 2 * bits // 8
    assert len(payload) == expected, f"mask payload is {len(payload)} B, expected {expected} at {bits} bits"


def test_deep_is_unshipped_but_states_a_real_test_accuracy(bundle):
    """Unshipped, yet measured exactly as the headline was -- both must be said.

    Calling it incomparable would throw away the comparison the board exists to
    make; calling it shipped would promote it by accident. The bundle has to
    carry both facts, because the widget renders captions from nothing else.
    """
    prov = bundle["provenance"]
    shipped = read_bundle("d2nn_weights.js")["provenance"]

    assert prov["shipped"] is False
    assert prov["caveat"], "an unshipped model must say why its number is not a headline"
    assert "not_scored_on" not in prov, (
        "this model was scored on the frozen test set; declaring otherwise would make "
        "the widget print 'not comparable' about the one honest comparison on the page."
    )
    assert prov["scored_on"] == shipped["scored_on"], "not the same measurement as the headline"
    assert prov["protocol"]["n_train"] == shipped["protocol"]["n_train"], "not the full training set"


def test_deep_shares_the_shipped_gallery(bundle):
    """Both models must be shown the same digits or the comparison is not one."""
    shipped = read_bundle("d2nn_weights.js")
    assert bundle["gallery_b64"] == shipped["gallery_b64"]
    assert bundle["gallery_labels"] == shipped["gallery_labels"]
    assert bundle["regions"] == shipped["regions"]
