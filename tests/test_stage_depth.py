"""The rules that make a deep stack legible (apps/web/d2nn_stage.js).

The 3D stage was built for the shipped 5-mask network and drew every plane on
every update. At 56 masks both halves of that break: 56 near-identical panels
communicate nothing, and rebuilding them on every pointer move makes the page
stutter. Three rules fix it, and all three are derived from the weight bundle
rather than from a mask count typed into the widget:

* **sampleMaskIndices** -- draw at most ``MAX_MASK_PANELS`` masks, always the
  first and the last, and never quietly.
* **defaultSubSteps** -- sub-step a hop only while the light spreads visibly
  across one.
* **mode** -- follow the pen for a shallow stack, wait for a button on a deep one.

The dangerous failure here is not a crash; it is a figure that looks right and
lies about which mask a visitor is looking at. That is what most of these check.

Requires Node on PATH; skips cleanly if absent.
"""
import json
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE_JS = os.path.join(HERE, "..", "apps", "web", "d2nn_stage.js").replace("\\", "/")

#: The two geometries that exist on the site, from their committed bundles.
SHIPPED_LAYERS, DEEP_LAYERS = 5, 56

node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node not on PATH; stage depth checks skipped")

_PROBE = """
const S = require('%s');
const W = (n_layers, separation) => ({
  n_layers, separation, wavelength: 532e-9, dx: 8e-6, n: 128,
});
process.stdout.write(JSON.stringify({
  MAX: S.MAX_MASK_PANELS,
  LIVE_MAX: S.LIVE_MAX_LAYERS,
  reachShipped: S.reachPerHopPx(W(5, 3e-3)),
  reachDeep: S.reachPerHopPx(W(56, 0.5263e-3)),
  subShipped: S.defaultSubSteps(W(5, 3e-3)),
  subDeep: S.defaultSubSteps(W(56, 0.5263e-3)),
  idx: Object.fromEntries([1, 2, 5, 6, 7, 9, 14, 28, 56, 80].map(
    (L) => [L, S.sampleMaskIndices(L)])),
}));
"""


@pytest.fixture(scope="module")
def probe():
    proc = subprocess.run([node, "-e", _PROBE % STAGE_JS], capture_output=True, text=True)
    assert proc.returncode == 0, f"node runner failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


# ------------------------------------------------------------------- sampling

def test_shallow_stacks_are_drawn_whole(probe):
    """Anything up to the limit keeps every plane -- including the shipped model.

    This is the compatibility half of the change: the published classifier page
    must look exactly as it did, so 5 masks must still mean 5 panels.
    """
    for L in (1, 2, 5, 6):
        assert probe["idx"][str(L)] == list(range(L)), f"{L} masks were sampled"
    assert probe["idx"][str(SHIPPED_LAYERS)] == [0, 1, 2, 3, 4]


def test_deep_stacks_are_sampled_within_the_budget(probe):
    """Never more panels than the limit, whatever the depth."""
    for L in (7, 9, 14, 28, 56, 80):
        got = probe["idx"][str(L)]
        assert len(got) <= probe["MAX"], f"{L} masks drew {len(got)} panels"
        assert got == sorted(set(got)), f"{L} masks gave repeated or unordered indices"


def test_sampling_keeps_the_ends_and_spreads_the_rest(probe):
    """First and last mask are always drawn, and the gaps stay even.

    The ends carry the two facts the figure is for -- what the digit looks like
    entering the stack and what it looks like leaving it -- and an uneven spread
    would suggest structure in the stack that is really an artifact of sampling.
    """
    for L in (9, 14, 28, 56, 80):
        got = probe["idx"][str(L)]
        assert got[0] == 0, f"{L} masks: first mask not drawn"
        assert got[-1] == L - 1, f"{L} masks: last mask not drawn"
        gaps = [b - a for a, b in zip(got, got[1:])]
        assert max(gaps) - min(gaps) <= 1, f"{L} masks: uneven spread {gaps}"


def test_every_index_is_a_real_mask(probe):
    """The label prints index+1, so an out-of-range index would name a mask that
    does not exist -- exactly the misreading the labels exist to prevent."""
    for L, got in probe["idx"].items():
        assert all(0 <= k < int(L) for k in got), f"{L} masks: {got} is out of range"


def test_the_deep_stack_says_it_is_sampled():
    """A sampled panel must never be labelled as if it were the whole stack.

    Checked in the source because it is a string, not a number: the label
    switches to 'mask K of N' the moment anything is left out, and the footer
    states the count. Both claims are load-bearing -- 'mask 23' alone would read
    as the 23rd of 6 drawn planes.
    """
    src = open(STAGE_JS, encoding="utf-8").read()
    assert "of ${W.n_layers}" in src, "sampled labels no longer name the true stack size"
    assert "masks drawn" in src, "the footer no longer says how many masks are drawn"


# ------------------------------------------------------------------ sub-steps

def test_substeps_reproduce_the_shipped_setting(probe):
    """The shipped stack keeps the 4 sub-hops it was tuned with.

    Its reach is 12.47 px per hop, so the derivation has to land on the value
    that was previously hardcoded, or this refactor silently changes a published
    figure.
    """
    assert probe["reachShipped"] == pytest.approx(12.47, abs=0.01)
    assert probe["subShipped"] == 4


def test_substeps_collapse_when_a_hop_barely_spreads(probe):
    """At 2.19 px per hop there is nothing to see between planes, so do not pay.

    This is what takes the deep stack's slice cost from 57*4 propagations to 57.
    """
    assert probe["reachDeep"] == pytest.approx(2.19, abs=0.01)
    assert probe["subDeep"] == 1


# ---------------------------------------------------------------------- mode

def test_live_for_the_shipped_stack_manual_for_a_deep_one(probe):
    """The mode threshold sits between the two geometries that actually exist."""
    assert SHIPPED_LAYERS <= probe["LIVE_MAX"], "the shipped stage would stop following the pen"
    assert DEEP_LAYERS > probe["LIVE_MAX"], "a 56-mask stage would chase every pointer move"


def test_the_first_result_draws_without_a_refresh():
    """Manual mode holds *updates*, not the first digit.

    An empty stage waiting for a button press reads as broken rather than as
    deliberate, so the guard is on `state.res` being absent.
    """
    src = open(STAGE_JS, encoding="utf-8").read()
    assert 'if (MODE === "live" || !state.res)' in src, (
        "setResult no longer renders the first result immediately"
    )
