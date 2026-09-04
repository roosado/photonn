"""The shared canvas primitives, tested once instead of nine times not at all.

``apps/web/plot.js`` holds what nine widgets each carried a private copy of: the
device-pixel-ratio clamp, the guarded canvas resize, the width observer, the
theme observer, the palette read, the colour ramps and the scalar-field
rasteriser.

The properties asserted here are the ones the private copies got wrong. The
resize guard is the headline: ``canvas.width = w`` throws the backing store away
and allocates a new one *even when the value is unchanged*, which is why the one
widget on this site that animates was reallocating a ~1440x860 surface on every
orbit frame. The theme observer is the other: the site's toggle writes
``data-theme`` on the root and never touches the OS preference, so a widget
listening only on ``matchMedia`` never hears it.
"""
import json
import shutil
import subprocess
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "plot_runner.js")

node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node not on PATH")


@pytest.fixture(scope="module")
def out():
    proc = subprocess.run([node, RUNNER], capture_output=True, text=True)
    assert proc.returncode == 0, f"plot runner failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def test_resize_is_guarded_on_the_dimensions_actually_changing(out):
    """The fix that 4b1e842 applied to two of three copies."""
    r = out["resize"]
    assert r["first"] is True, "a first sizing must resize"
    assert r["second"] is False, (
        "re-sizing to the same dimensions must not touch canvas.width -- that is "
        "the reallocation this module exists to prevent"
    )
    assert r["third"] is True, "a genuine dimension change must still resize"


def test_device_pixel_ratio_is_clamped(out):
    s = out["scale"]
    assert s["clamped"] == s["max"] == 2, "a dpr-3 screen must be capped at 2"
    assert s["passed"] == 1.5, "a ratio under the cap passes through unchanged"


def test_fit_sizes_backing_store_and_re_establishes_the_transform(out):
    f = out["fit"]
    assert f["backing"] == [f["W"] * f["dpr"], f["H"] * f["dpr"]]
    assert f["styleHeight"] == f"{f['H']}px"
    assert f["stableAcrossCalls"], "a second fit at the same width must not resize"
    assert f["transforms"] == 2, (
        "setTransform must be applied on every fit -- both of them here. After a "
        "real resize it is required (a resize clears the context); after a "
        "skipped one it is a harmless no-op. Applying it only when the resize "
        "happened is the bug this shape avoids."
    )


def test_fit_to_takes_both_dimensions(out):
    assert out["fitTo"]["backing"] == [600, 240]      # 300x120 CSS at dpr 2
    assert out["fitTo"]["styleHeight"] == "120px"


def test_the_ramps_are_built_once_and_correctly(out):
    lut = out["lut"]
    assert lut["length"] == 256 * 3
    assert lut["firstIsBlack"] == [0, 0, 4]
    assert lut["lastIsBright"] == [252, 255, 164]
    assert all(lut["phaseJoins"]), (
        "the phase ramp must join end to end: -pi and +pi are the same setting of "
        "the same mask, and a ramp that does not close draws a false seam"
    )


def test_raster_produces_an_n_by_n_bitmap(out):
    r = out["raster"]
    assert (r["width"], r["height"]) == (4, 4)
    assert r["put"], "raster must actually write its image data"
    assert out["rasterCyclicRuns"]


def test_theme_changes_are_watched_on_the_attribute_the_toggle_writes(out):
    t = out["theme"]
    assert t["observesRoot"], "the observer must watch document.documentElement"
    assert t["filter"] == ["data-theme"], (
        "matchMedia alone never fires for this site's own toggle"
    )


def test_a_stylesheet_is_injected_once_per_id(out):
    s = out["injectStyle"]
    assert s["created"] and s["notOverwritten"]
