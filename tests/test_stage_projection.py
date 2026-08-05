"""The 3D stage's projection and canvas transform (apps/web/d2nn_stage.js).

The stage draws each plane with a single affine transform built from an
orthographic basis. If that basis is wrong the whole figure is a plausible-looking
lie, so it gets the same treatment as the physics: checked, not eyeballed.

Most checks run the shipped JavaScript under Node and skip cleanly without it. The
last one reads the source instead, because the bug it guards against only shows up
on a canvas at devicePixelRatio > 1 -- which Node cannot provide.
"""
import json
import math
import os
import re
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "stage_projection_runner.js")
STAGE_JS = os.path.join(HERE, "..", "apps", "web", "d2nn_stage.js")

node = shutil.which("node")
needs_node = pytest.mark.skipif(node is None, reason="node not on PATH; stage checks skipped")


@pytest.fixture(scope="module")   # only requested by @needs_node tests, so never runs without it
def rows():
    proc = subprocess.run([node, RUNNER], capture_output=True, text=True)
    assert proc.returncode == 0, f"node runner failed:\n{proc.stderr}"
    return json.loads(proc.stdout)["rows"]


def at(rows, theta, phi):
    for r in rows:
        if r["theta"] == theta and r["phi"] == phi:
            return r
    raise KeyError(f"no row for theta={theta}, phi={phi}")


@needs_node
def test_depth_axis_collapses_head_on(rows):
    """At theta = phi = 0 the optical axis points straight at the viewer.

    Every panel then lands exactly on top of the others and the figure degenerates
    to the single-plane view -- the cheapest decisive check the basis vectors are
    the right way round.
    """
    r = at(rows, 0, 0)
    assert r["eZ"] == [0.0, 0.0]
    assert r["eX"] == [1.0, 0.0]
    assert r["eY"] == [0.0, 1.0]


@needs_node
def test_basis_is_a_real_orthographic_projection(rows):
    """The projected columns of a rotation keep sum-of-squared-norms equal to 2.

    That identity is what survives dropping the view dimension; a basis assembled
    from mismatched sines would break it while still drawing something.
    """
    for r in rows:
        assert r["sumSq"] == pytest.approx(2.0, rel=1e-12), (
            f"theta={r['theta']}, phi={r['phi']}: sum of squared norms {r['sumSq']}"
        )


@needs_node
def test_in_plane_vertical_axis_stays_vertical(rows):
    """The plane's y axis never acquires a horizontal component.

    Panels are drawn by ``drawImage`` under this basis, so a non-zero eX component
    on eY would shear every field image.
    """
    for r in rows:
        assert r["eY"][0] == 0.0
        assert r["eY"][1] == pytest.approx(math.cos(math.radians(r["phi"])), rel=1e-12)


@needs_node
def test_increasing_depth_moves_toward_the_viewer(rows):
    """Deeper planes must draw lower on screen, which is what makes ascending-z
    painting the exact painter's order for the default view."""
    r = at(rows, 34, 19)
    assert r["eZ"][0] > 0, "deeper planes should move right"
    assert r["eZ"][1] > 0, "deeper planes should move down, i.e. nearer the viewer"


def test_plane_bitmaps_compose_with_the_dpr_transform():
    """`drawPlane` must use ctx.transform, never ctx.setTransform.

    Regression guard for a real bug. ``draw()`` puts a devicePixelRatio scale on
    the context and every other mark -- plates, outlines, detector boxes, labels --
    is a plain path drawn under it. ``setTransform`` *replaces* the matrix, so the
    light bitmaps lost that scale and were drawn 1/dpr of the way toward the canvas
    origin while the rig stayed put. Measured on the shipped page: the light's span
    collapsed 388 -> 194 -> 129 CSS px at dpr 1 -> 2 -> 3.

    It is invisible at dpr 1, which is why it survived review and every desktop
    screenshot, and it cannot be caught under Node -- there is no canvas. Hence a
    source check: the only legitimate ``setTransform`` in this file is the one that
    establishes the dpr scale in the first place.
    """
    src = open(STAGE_JS, encoding="utf-8").read()

    body = re.search(r"function drawPlane\(.*?\n    \}", src, re.S)
    assert body, "drawPlane not found -- update this guard if it was renamed"
    assert "ctx.transform(" in body.group(0), "drawPlane no longer composes onto the CTM"
    assert "ctx.setTransform(" not in body.group(0), (
        "drawPlane calls ctx.setTransform, which discards the devicePixelRatio scale "
        "and shrinks the light by 1/dpr away from the panels that frame it"
    )

    # Strip comments so prose about the trap does not count as a call site.
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", src, flags=re.S)
    setters = re.findall(r"ctx\.setTransform\(([^)]*)\)", code)
    assert setters == ["dpr, 0, 0, dpr, 0, 0"], (
        f"expected exactly one setTransform (the dpr scale); found {setters}"
    )
