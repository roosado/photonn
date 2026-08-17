"""Layout invariants for the seven error-mechanism widgets.

``apps/web/errors.js`` draws one widget per fabrication error source, all seven on
/tolerance: six against the diffractive stack, and ``mesh`` against the MZI chip in
the section that compares the two. Two of its layout properties are load-bearing
for whether the widget teaches anything at all, and neither is visible to any other
test:

* **The triptychs must not wrap.** Five of the seven (crosstalk, phase,
  wavelength, quant, mesh) are a before/after/difference row. Sweeping the slider
  only shows you
  the mechanism if all three panels are on screen at once. They used to be
  ``flex: 1 1 150px; min-width: 132px``, which needs 420 px of row -- more than a
  phone has -- so the third panel wrapped, each panel then grew to the full
  column width, and the reader got one enormous panel at a time and no
  comparison.

* **A plot's bitmap aspect must match the aspect it is displayed at.** The two
  plot widgets (loss, detector) drew into a fixed 420-unit space and let
  ``width:100%`` stretch the result to whatever the column happened to be. That
  scales x without scaling y, so the chart and its labels came out flattened on a
  desktop and stretched tall on a phone.

Neither can be caught in the driven browser: that tab is always hidden, so it
never lays anything out. So the JS half is mounted under Node against a DOM
stand-in (``tests/error_widget_runner.js``) and the CSS half is asserted over the
stylesheet source, which is where the wrapping rule actually lives.
"""
import json
import os
import re
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "error_widget_runner.js")
ERRORS_JS = os.path.join(HERE, "..", "apps", "web", "errors.js")

#: Widgets whose three panels are meant to be read side by side.
TRIPTYCHS = ("crosstalk", "phase", "wavelength", "quant", "mesh")
#: Widgets that draw a single self-sized plot.
PLOTS = ("loss", "detector")

node = shutil.which("node")


@pytest.fixture(scope="module")
def source():
    return open(ERRORS_JS, encoding="utf-8").read()


@pytest.fixture(scope="module")
def css(source):
    body = re.search(r"const CSS = `(.*?)`;", source, re.S)
    assert body, "could not find the CSS template literal in errors.js"
    return body.group(1)


@pytest.fixture(scope="module")
def out():
    if node is None:
        pytest.skip("node not on PATH")
    proc = subprocess.run([node, RUNNER], capture_output=True, text=True)
    assert proc.returncode == 0, f"error widget runner failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


# ----------------------------------------------------------------- stylesheet

def test_the_panel_row_never_wraps(css):
    """A wrapped triptych is the bug: one panel at a time, nothing to compare."""
    row = re.search(r"\.ex-panes\{([^}]*)\}", css)
    assert row, ".ex-panes rule not found"
    assert "flex-wrap" not in row.group(1), (
        "the panel row must not wrap -- all three panels have to stay on screen "
        "while the slider moves, which is the entire point of the widget"
    )


def test_panels_may_shrink_below_their_content(css):
    """Without min-width:0 a flex item refuses to shrink and overflows instead."""
    pane = re.search(r"\.ex-pane\{([^}]*)\}", css)
    assert pane, ".ex-pane rule not found"
    body = pane.group(1)
    assert "min-width:0" in body, (
        "panels need min-width:0 or they cannot shrink to share a narrow row"
    )
    assert re.search(r"flex:\s*1\s+1\s+0", body), (
        "panels need a zero flex-basis so three of them split the row evenly"
    )


def test_the_wide_plot_pane_is_capped(css):
    """Full-bleed is wrong for a 150 px-tall chart on a 1000 px column."""
    wide = re.search(r"\.ex-pane\.ex-wide\{([^}]*)\}", css)
    assert wide, ".ex-pane.ex-wide rule not found"
    assert "max-width" in wide.group(1)


def test_no_widget_sets_layout_with_an_inline_style(source):
    """Layout belongs in the stylesheet, where the media query can reach it.

    The wide pane used to be set as ``c.parentElement.style.flex = "1 1 100%"``,
    which no media query can override.
    """
    assert ".style.flex" not in source, (
        "set a class and let the stylesheet decide, so narrow-screen rules apply"
    )


# ------------------------------------------------------------------- mounting

def test_every_kind_mounts_at_every_width(out):
    """Seven widgets, three widths, two pixel ratios, no exceptions thrown.

    The runner keeps its own list of kinds, so this is also the check that a new
    widget was added to it: without that, the seventh kind ships untested and
    nothing here would have noticed.
    """
    assert set(out["kinds"]) == set(TRIPTYCHS) | set(PLOTS)
    keys = [k for k in out if k != "kinds"]
    assert len(keys) == 7 * 3 * 2, f"expected 42 mounted widgets, got {len(keys)}"


def test_the_runner_covers_every_kind_the_widget_file_defines(source):
    """The list in the runner must not fall behind ``KINDS`` in errors.js."""
    defined = set(re.findall(r"KINDS\.(\w+) = function", source))
    assert defined == set(TRIPTYCHS) | set(PLOTS), (
        f"errors.js defines {sorted(defined)}; this file tests "
        f"{sorted(set(TRIPTYCHS) | set(PLOTS))}. A kind missing here is a kind "
        "that ships with no layout test at all."
    )


def test_triptychs_keep_three_panels_on_one_row(out):
    for key, r in out.items():
        if key == "kinds" or r["kind"] not in TRIPTYCHS:
            continue
        assert r["panes"] == 3, f"{key}: {r['panes']} panels, expected 3"
        assert r["perRow"] == 3, (
            f"{key}: only {r['perRow']} of 3 panels fit on a row at {r['width']} px, "
            "so the rest wrap and the comparison is lost"
        )
        widths = {c["shownW"] for c in r["canvases"]}
        assert len(widths) == 1, f"{key}: panels are not equal width -- {widths}"
        # Three equal panels plus two gaps must fit the row they were given.
        assert r["canvases"][0]["shownW"] * 3 < r["width"], (
            f"{key}: three panels do not fit in {r['width']} px"
        )


def test_panel_canvases_are_square_and_left_to_scale_themselves(out):
    """Square bitmap + `height:auto` is what keeps a panel undistorted."""
    for key, r in out.items():
        if key == "kinds" or r["kind"] not in TRIPTYCHS:
            continue
        for c in r["canvases"]:
            assert c["bitmapW"] == c["bitmapH"], f"{key}: {c['bitmapW']}x{c['bitmapH']}"
            assert c["styleH"] is None, (
                f"{key}: an inline height on a square panel overrides height:auto "
                "and distorts it"
            )


def test_plot_bitmaps_match_the_aspect_they_are_displayed_at(out):
    """The regression this file exists for.

    A plot is drawn in CSS-pixel units and then displayed at the pane's real
    width. If the bitmap's aspect and the displayed aspect disagree, the browser
    scales one axis and not the other, and every bar, gap and glyph is stretched
    by that ratio.
    """
    for key, r in out.items():
        if key == "kinds" or r["kind"] not in PLOTS:
            continue
        for c in r["canvases"]:
            assert c["styleH"], f"{key}: a plot must pin its own height"
            drawn = c["bitmapW"] / c["bitmapH"]
            shown = c["shownW"] / c["styleH"]
            assert abs(drawn - shown) < 0.01, (
                f"{key}: bitmap {c['bitmapW']}x{c['bitmapH']} (aspect {drawn:.3f}) "
                f"shown at {c['shownW']:.0f}x{c['styleH']} (aspect {shown:.3f}) -- "
                "one axis is being stretched"
            )


def test_plots_track_the_device_pixel_ratio_without_changing_shape(out):
    """A 2x screen doubles the bitmap and leaves the CSS box alone."""
    for kind in PLOTS:
        for width in (300, 480, 1042):
            one = out[f"{kind}@{width}x1"]["canvases"][0]
            two = out[f"{kind}@{width}x2"]["canvases"][0]
            assert two["bitmapW"] == one["bitmapW"] * 2
            assert two["bitmapH"] == one["bitmapH"] * 2
            assert two["styleH"] == one["styleH"]
            assert two["shownW"] == one["shownW"]


def test_a_plot_is_readable_rather_than_a_letterbox(out):
    """Height follows width between bounds, so neither extreme goes silly."""
    for kind in PLOTS:
        narrow = out[f"{kind}@300x1"]["canvases"][0]
        wide = out[f"{kind}@1042x1"]["canvases"][0]
        assert wide["shownW"] <= 640, "a wide plot must stay capped"
        assert wide["styleH"] > narrow["styleH"], (
            "a wider plot should get taller, or it degenerates into a letterbox"
        )
        for c in (narrow, wide):
            assert 1.5 <= c["shownW"] / c["styleH"] <= 4.0, (
                f"{kind}: aspect {c['shownW'] / c['styleH']:.2f} is outside the "
                "range these charts read well at"
            )
