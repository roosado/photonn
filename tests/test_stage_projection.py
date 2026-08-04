"""The 3D stage's projection basis (apps/web/d2nn_stage.js).

The stage draws each plane with a single affine transform built from an
orthographic basis. If that basis is wrong the whole figure is a plausible-looking
lie, so it gets the same treatment as the physics: checked, not eyeballed.

Runs the shipped JavaScript under Node; skips cleanly if Node is absent.
"""
import json
import math
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "stage_projection_runner.js")

node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node not on PATH; stage checks skipped")


@pytest.fixture(scope="module")
def rows():
    proc = subprocess.run([node, RUNNER], capture_output=True, text=True)
    assert proc.returncode == 0, f"node runner failed:\n{proc.stderr}"
    return json.loads(proc.stdout)["rows"]


def at(rows, theta, phi):
    for r in rows:
        if r["theta"] == theta and r["phi"] == phi:
            return r
    raise KeyError(f"no row for theta={theta}, phi={phi}")


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


def test_basis_is_a_real_orthographic_projection(rows):
    """The projected columns of a rotation keep sum-of-squared-norms equal to 2.

    That identity is what survives dropping the view dimension; a basis assembled
    from mismatched sines would break it while still drawing something.
    """
    for r in rows:
        assert r["sumSq"] == pytest.approx(2.0, rel=1e-12), (
            f"theta={r['theta']}, phi={r['phi']}: sum of squared norms {r['sumSq']}"
        )


def test_in_plane_vertical_axis_stays_vertical(rows):
    """The plane's y axis never acquires a horizontal component.

    Panels are drawn by ``drawImage`` under this basis, so a non-zero eX component
    on eY would shear every field image.
    """
    for r in rows:
        assert r["eY"][0] == 0.0
        assert r["eY"][1] == pytest.approx(math.cos(math.radians(r["phi"])), rel=1e-12)


def test_increasing_depth_moves_toward_the_viewer(rows):
    """Deeper planes must draw lower on screen, which is what makes ascending-z
    painting the exact painter's order for the default view."""
    r = at(rows, 34, 19)
    assert r["eZ"][0] > 0, "deeper planes should move right"
    assert r["eZ"][1] > 0, "deeper planes should move down, i.e. nearer the viewer"
