"""The contract between an exported model bundle and the browser widgets.

Two trained models now share one page, and a third will arrive when a promotion
lands. What keeps that from becoming a maintenance trap is a rule: **a widget
holds no model metadata.** Every caption on ``apps/web/d2nn_compare.js`` -- the
column title, the accuracy, what it was scored on, the unshipped caveat, the
orange border -- is rendered from the bundle's own ``provenance`` block, so
promoting a model is regenerating a bundle and nothing else.

The rule is invisible at runtime: a widget that quietly hardcodes last quarter's
accuracy looks perfect and is wrong. That has already happened once in this repo
(``77b8a31``), so it is asserted here rather than remembered.

Three things are checked:

* every bundle carries the fields the captions read;
* no widget contains either model's accuracy as a literal;
* the caption builders actually change when the provenance changes -- including
  for a *promoted* bundle that has never existed, which is the case no committed
  file can exercise.

The caption checks need Node and skip cleanly without it; the rest are plain
file reads and always run.
"""
import json
import os
import re
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "apps", "web")

BUNDLES = ["d2nn_weights.js", "d2nn_sweep_weights.js", "d2nn_deep_weights.js"]
WIDGETS = ["d2nn_compare.js", "digit_source.js"]

#: Fields d2nn_compare.js reads off every model it draws a column for.
REQUIRED = ("label", "accuracy", "scored_on", "shipped", "protocol")
REQUIRED_PROTOCOL = ("n_train", "epochs", "seed")

node = shutil.which("node")
needs_node = pytest.mark.skipif(node is None, reason="node not on PATH")


def read_bundle(name):
    """Parse a generated weights bundle's payload out of its IIFE."""
    text = open(os.path.join(WEB, name), encoding="utf-8").read()
    body = re.search(r"var W = (\{.*?\});\n", text, re.S)
    assert body, f"{name} is not a generated bundle"
    return json.loads(body.group(1))


# --------------------------------------------------------------------- bundles

@pytest.mark.parametrize("name", BUNDLES)
def test_bundle_carries_provenance(name):
    """Both bundles, not only the candidate's.

    The asymmetry this guards against is the original one: ``export_sweep_web``
    wrote a provenance block and ``export_d2nn_web`` did not, which is precisely
    why the shipped model's accuracy was a literal in JavaScript.
    """
    prov = read_bundle(name).get("provenance")
    assert prov, f"{name} has no provenance block"
    for key in REQUIRED:
        assert key in prov, f"{name} provenance lacks {key!r}"
    assert isinstance(prov["shipped"], bool), "shipped must be a bool, not a truthy string"
    assert 0.0 < prov["accuracy"] <= 1.0, f"{name} accuracy is not a fraction"
    for key in REQUIRED_PROTOCOL:
        assert key in prov["protocol"], f"{name} protocol lacks {key!r}"


def test_exactly_one_bundle_claims_to_be_shipped():
    """`Not comparable to X` needs an unambiguous X, and the site needs one truth."""
    shipped = [n for n in BUNDLES if read_bundle(n)["provenance"]["shipped"]]
    assert shipped == ["d2nn_weights.js"], f"expected one shipped model, got {shipped}"


# --------------------------------------------------------------------- widgets

@pytest.mark.parametrize("widget", WIDGETS)
def test_no_accuracy_is_written_into_a_widget(widget):
    """The failure this step exists to prevent, stated as an assertion.

    Checked against the numbers the bundles actually carry, at the precisions a
    caption would plausibly be written at -- so it keeps working after a
    retrain, when the literal to look for is a different one.
    """
    src = open(os.path.join(WEB, widget), encoding="utf-8").read()
    for name in BUNDLES:
        acc = read_bundle(name)["provenance"]["accuracy"]
        for places in (2, 3, 4):
            literal = f"{acc:.{places}f}"
            assert literal not in src, (
                f"{widget} hardcodes {literal}, the accuracy of {name}. Captions must be "
                "rendered from the bundle's provenance block, never typed into a widget."
            )


# ------------------------------------------------------------------- captions

_CAPTIONS = """
const C = require('%s');
const cases = %s;
process.stdout.write(JSON.stringify(
  cases.map(c => C.noteFor(c.prov, c.reference))));
"""


def captions(cases):
    """Render d2nn_compare.js's verdict captions for a list of provenance blocks."""
    mod = os.path.join(WEB, "d2nn_compare.js").replace("\\", "/")
    proc = subprocess.run([node, "-e", _CAPTIONS % (mod, json.dumps(cases))],
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"node runner failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


@needs_node
def test_shipped_caption_states_its_accuracy_and_where_it_came_from():
    prov = read_bundle("d2nn_weights.js")["provenance"]
    note, = captions([{"prov": prov, "reference": None}])

    assert f"{prov['accuracy']:.4f}" in note
    assert prov["scored_on"] in note
    assert f"{prov['protocol']['epochs']} epochs" in note
    assert "Not shipped" not in note, "the shipped model must not carry the unshipped caveat"


@needs_node
def test_candidate_caption_refuses_to_be_read_as_a_test_accuracy():
    """The number beside an operable model invites being quoted; this disarms it."""
    shipped = read_bundle("d2nn_weights.js")["provenance"]
    cand = read_bundle("d2nn_sweep_weights.js")["provenance"]
    note, = captions([{"prov": cand, "reference": shipped}])

    assert "Not shipped" in note
    assert f"{cand['accuracy']:.4f}" in note
    assert cand["caveat"] in note
    # The headline it must not be confused with is named, and comes from the
    # shipped bundle rather than from a literal in the widget.
    assert f"Not comparable to {shipped['accuracy']:.4f}" in note


@needs_node
def test_the_deep_candidate_is_not_disclaimed_out_of_its_own_comparison():
    """Unshipped does not mean incomparable, and the caption must tell them apart.

    The 56-mask model was scored on the same frozen test set, at the same
    training budget, as the shipped one -- so ``Not comparable to 0.7990`` beside
    it would be false, and would disclaim away the single honest comparison the
    board is built to show. The distinguishing fact lives in the bundles (a
    ranking run declares ``not_scored_on``; this one does not), never in the
    widget.
    """
    shipped = read_bundle("d2nn_weights.js")["provenance"]
    deep = read_bundle("d2nn_deep_weights.js")["provenance"]
    ranked = read_bundle("d2nn_sweep_weights.js")["provenance"]
    note, = captions([{"prov": deep, "reference": shipped}])

    assert "Not shipped" in note, "an unshipped model still says so"
    assert f"{deep['accuracy']:.4f}" in note
    assert deep["caveat"] in note
    assert "Not comparable" not in note
    assert f"{shipped['accuracy']:.4f}" in note, "the number it *is* comparable to is named"
    # And the ranking candidate on the same board still is disclaimed.
    ranked_note, = captions([{"prov": ranked, "reference": shipped}])
    assert f"Not comparable to {shipped['accuracy']:.4f}" in ranked_note


@needs_node
def test_a_promoted_bundle_changes_the_caption_with_no_javascript_edit():
    """The acceptance test: fabricate a promotion and watch the captions follow.

    No committed bundle can exercise this -- the candidate has never been
    promoted -- so the provenance is synthetic on purpose. The numbers below are
    invented *as test fixtures*, never as physics or as results.
    """
    promoted = {
        "label": "Promoted",
        "accuracy": 0.8642,
        "scored_on": "the frozen 2,000-image MNIST test set",
        "shipped": True,
        "protocol": {"n_train": 60000, "epochs": 40, "seed": 1},
    }
    note, = captions([{"prov": promoted, "reference": None}])

    assert "0.8642" in note
    assert promoted["scored_on"] in note
    # Promotion removes the caveat by itself; nothing in the widget knows which
    # model was promoted.
    assert "Not shipped" not in note
    assert "Not comparable" not in note


@needs_node
def test_a_bundle_without_a_protocol_still_renders():
    """A caption may lose detail when a field is absent, but must not break."""
    note, = captions([{"prov": {"label": "Bare", "accuracy": 0.5, "shipped": True},
                       "reference": None}])
    assert "0.5000" in note
    assert "undefined" not in note and "NaN" not in note


# --------------------------------------------------------------- digit source

_SOURCE_PROBE = """
const SRC = require('%s');
const NET = require('%s');

// A synthetic stroke: a short vertical bar in one corner of a 196x196 pad --
// deliberately off-centre and the wrong size, which is exactly what MNIST
// normalisation is for.
const PAD = 196;
const gray = new Float64Array(PAD * PAD);
for (let y = 8; y < 60; y++) for (let x = 12; x < 22; x++) gray[y * PAD + x] = 1;

const d = NET.normalizeDrawn(gray, PAD);
const D = NET.DIGIT;
let sum = 0, cx = 0, cy = 0;
for (let j = 0; j < D; j++) for (let i = 0; i < D; i++) {
  const v = d[j * D + i];
  sum += v; cx += v * i; cy += v * j;
}
process.stdout.write(JSON.stringify({
  surface: Object.keys(SRC),
  mountIsFunction: typeof SRC.mount === 'function',
  length: d.length, size: D, sum,
  cx: sum > 0 ? cx / sum : null,
  cy: sum > 0 ? cy / sum : null,
}));
"""


@needs_node
def test_digit_source_exports_a_mountable_widget():
    src = os.path.join(WEB, "digit_source.js").replace("\\", "/")
    d2nn = os.path.join(WEB, "d2nn.js").replace("\\", "/")
    proc = subprocess.run([node, "-e", _SOURCE_PROBE % (src, d2nn)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"node runner failed:\n{proc.stderr}"
    out = json.loads(proc.stdout)

    assert out["surface"] == ["mount"]
    assert out["mountIsFunction"]

    # The normalisation the source depends on: a corner stroke comes back as a
    # 28x28 digit with its ink centred, which is what makes a drawing look to
    # the trained masks like the digits they were trained on.
    assert out["length"] == out["size"] ** 2 == 28 * 28
    assert out["sum"] > 0, "the stroke vanished"
    assert abs(out["cx"] - 13.5) < 1.5, f"ink not centred in x (cx={out['cx']:.2f})"
    assert abs(out["cy"] - 13.5) < 1.5, f"ink not centred in y (cy={out['cy']:.2f})"
