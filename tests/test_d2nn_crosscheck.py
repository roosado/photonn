"""Cross-check the browser D2NN against the trained torch model.

The demo page runs the trained diffractive network's forward pass live in
JavaScript (apps/web/d2nn.js), so its predictions must be the trained model's
predictions -- not merely similar ones. This test runs the same JS the page ships
under Node against reference logits generated from exports/d2nn_phase2.pt and
asserts:

1. every predicted label matches torch exactly (the claim the demo makes);
2. logits agree to < 1e-3 (float32 masks + float64 JS arithmetic);
3. the JS bilinear resize matches torch's align_corners=False convention, the one
   place a half-pixel error would silently poison every prediction;
4. the bundled geometry still matches the handoff it was generated from.

Fixture and weights are produced by ``python -m apps.export_d2nn_web``; re-run it
if the trained model changes. Requires Node on PATH; skips cleanly if absent.
"""
import json
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "d2nn_crosscheck_runner.js")
FIXTURE = os.path.join(HERE, "fixtures", "d2nn_reference.json")
WEIGHTS = os.path.join(HERE, "..", "apps", "web", "d2nn_weights.js")

LOGIT_TOL = 1e-3
RESIZE_TOL = 1e-5

node = shutil.which("node")


@pytest.fixture(scope="module")
def results():
    assert os.path.exists(FIXTURE), "run `python -m apps.export_d2nn_web` first"
    assert os.path.exists(WEIGHTS), "run `python -m apps.export_d2nn_web` first"
    proc = subprocess.run([node, RUNNER], capture_output=True, text=True)
    assert proc.returncode == 0, f"node runner failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def fixture_data():
    with open(FIXTURE, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.skipif(node is None, reason="node not on PATH; browser D2NN cross-check skipped")
def test_predictions_match_torch(results):
    """The browser network predicts exactly what the trained model predicts."""
    assert results["cases"], "no cases in fixture"
    mismatched = [c for c in results["cases"] if c["predJs"] != c["predRef"]]
    assert not mismatched, (
        f"{len(mismatched)}/{len(results['cases'])} predictions differ from torch: "
        + ", ".join(f"digit {c['index']}: JS {c['predJs']} vs torch {c['predRef']}"
                    for c in mismatched[:5])
    )


@pytest.mark.skipif(node is None, reason="node not on PATH; browser D2NN cross-check skipped")
def test_logits_match_torch(results):
    """Class logits agree with torch to within float32-mask precision."""
    worst = max(results["cases"], key=lambda c: c["maxLogitErr"])
    assert worst["maxLogitErr"] < LOGIT_TOL, (
        f"digit {worst['index']}: max abs logit error {worst['maxLogitErr']:.2e} "
        f"exceeds {LOGIT_TOL:.0e}"
    )


@pytest.mark.skipif(node is None, reason="node not on PATH; browser D2NN cross-check skipped")
def test_resize_matches_torch_convention(results):
    """The JS bilinear resize reproduces torch's align_corners=False canvases."""
    assert results["resize"], "no resize cases in fixture"
    worst = max(results["resize"], key=lambda c: c["maxErr"])
    assert worst["maxErr"] < RESIZE_TOL, (
        f"digit {worst['index']}: input-canvas error {worst['maxErr']:.2e} exceeds "
        f"{RESIZE_TOL:.0e} -- the JS resize convention drifted from torch's"
    )


@pytest.mark.skipif(node is None, reason="node not on PATH; browser D2NN cross-check skipped")
def test_bundled_geometry_matches_handoff(results, fixture_data):
    """The shipped weights describe the same optical system as the handoff."""
    assert results["geometry"] == pytest.approx(fixture_data["geometry"]), (
        "apps/web/d2nn_weights.js is stale relative to the fixture; re-run "
        "`python -m apps.export_d2nn_web`"
    )


@pytest.mark.skipif(node is None, reason="node not on PATH; browser D2NN cross-check skipped")
def test_gallery_is_honest(results):
    """The shipped gallery is neither cherry-picked perfect nor broken.

    The exporter deliberately includes digits the model gets wrong, so a gallery
    that scores 100% means the honest-failure selection silently stopped working.
    """
    g = results["gallery"]
    assert 0 < g["correct"] < g["total"], (
        f"gallery accuracy {g['correct']}/{g['total']} -- expected a mix of "
        "correct and misclassified digits"
    )
