"""The front page's interference widget: does it draw the identity it claims?

``apps/web/interfere.js`` exists to make one sentence operable: two waves of equal
brightness, held apart by a phase, reinforce or cancel. Everything downstream of
that sentence on this site depends on it, so the widget has to be right in two
senses that nothing else in the suite can see.

* **The picture is the physics.** ``samples()`` computes ``cos(kx) + cos(kx - d)``
  pointwise and the canvas draws that array; ``envelope()`` is the closed form
  ``2cos(d/2)`` the caption and the readout quote. If those two ever part company
  the front page is illustrating something that is not true, confidently. Here
  they are checked against each other to 1e-12, including the two endpoints by
  name: in step the sum stands at twice one wave, exactly opposed it is flat.

* **The canvas is the size it is displayed at.** A bitmap whose aspect does not
  match its display aspect is silently stretched in one axis. ``errors.js``
  shipped that bug in both of its plot widgets, and it is invisible to every
  other kind of test: the driven Chrome tab is always hidden, so it never lays
  anything out. ``tests/interference_runner.js`` mounts the widget against a DOM
  stand-in at three widths and two pixel ratios, and reads the plot cap out of
  the widget's own stylesheet rather than restating it.

The third thing asserted here is a promise to the reader rather than a fact about
the code: the widget must be able to reach **total** cancellation, and must say
so in words when it does. A version that bottomed out at "nearly dark" would be a
different, weaker claim than the one the paragraph above it makes.
"""
import json
import os
import re
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "interference_runner.js")
WIDGET_JS = os.path.join(HERE, "..", "apps", "web", "interfere.js")

PI = 3.141592653589793

node = shutil.which("node")


@pytest.fixture(scope="module")
def source():
    return open(WIDGET_JS, encoding="utf-8").read()


@pytest.fixture(scope="module")
def css(source):
    body = re.search(r"const CSS = `(.*?)`;", source, re.S)
    assert body, "could not find the CSS template literal in interfere.js"
    return body.group(1)


@pytest.fixture(scope="module")
def out():
    if node is None:
        pytest.skip("node not on PATH")
    # utf-8 explicitly: the readout prints a lambda, and the Windows default
    # locale encoding turns it into mojibake that the assertions then chase.
    proc = subprocess.run([node, RUNNER], capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, f"interference runner failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def at(out, dphi):
    return out["physics"][f"{dphi:.6f}"]


# -------------------------------------------------------------------- physics

@pytest.mark.parametrize("dphi", [0, PI / 4, PI / 2, PI, 3 * PI / 2, 2 * PI])
def test_the_drawn_sum_is_the_closed_form(out, dphi):
    """cos(kx) + cos(kx - d) == 2cos(d/2)cos(kx - d/2), pointwise."""
    row = at(out, dphi)
    assert row["maxErr"] < 1e-12, (
        f"at d={dphi:.3f} the sampled sum departs from the closed form by "
        f"{row['maxErr']:.2e}; the widget is drawing something other than the "
        "identity its caption states"
    )


def test_in_step_the_waves_add_to_twice_one_wave(out):
    row = at(out, 0)
    assert abs(row["envelope"] - 2.0) < 1e-12
    assert abs(row["peak"] - 2.0) < 1e-12, "the drawn curve never reaches 2x"
    assert abs(row["brightness"] - 1.0) < 1e-12


def test_exactly_opposed_the_light_goes_out_completely(out):
    """Not "nearly dark". The paragraph this widget serves says it cancels."""
    row = at(out, PI)
    assert abs(row["envelope"]) < 1e-12
    assert abs(row["peak"]) < 1e-12, (
        f"the drawn sum still peaks at {row['peak']:.2e} at half a wavelength; "
        "the widget cannot show cancellation, which is the one thing it is for"
    )
    assert abs(row["brightness"]) < 1e-12


def test_a_quarter_wave_is_half_the_light(out):
    """The fact worth a reader's attention: brightness is amplitude squared."""
    row = at(out, PI / 2)
    assert abs(row["brightness"] - 0.5) < 1e-12
    assert abs(row["envelope"] - 2 ** 0.5) < 1e-12


def test_a_full_wavelength_is_back_in_step(out):
    """The slider's far end has to land back where it started, not somewhere new."""
    assert abs(abs(at(out, 2 * PI)["envelope"]) - 2.0) < 1e-12
    assert abs(at(out, 2 * PI)["brightness"] - 1.0) < 1e-12


# -------------------------------------------------------------------- readout

def test_the_readout_is_derived_from_the_same_numbers(out):
    """The printed brightness is formatted from brightness(), not computed twice."""
    for dphi, expect in ((0, "1.00"), (PI / 2, "0.50"), (PI, "0.00")):
        read = out["readout"][f"{dphi:.6f}"]["read"]
        assert f"<b>{expect}</b>" in read, f"at d={dphi:.3f} the meter reads {read!r}"


def test_the_swatch_emits_what_it_prints(out):
    """Black when the light is gone, white when it is doubled, and gamma between.

    A display is gamma-encoded, so a linear ``255*b`` would render half the light
    as a patch that looks like a fifth of it and quietly contradict the number
    printed beside it.
    """
    swatches = {k: v["swatch"] for k, v in out["readout"].items()}
    assert swatches[f"{PI:.6f}"] == "rgb(0,0,0)"
    assert swatches[f"{0:.6f}"] == "rgb(255,255,255)"
    half = int(re.search(r"rgb\((\d+)", swatches[f"{PI / 2:.6f}"]).group(1))
    assert 180 <= half <= 192, (
        f"half the light rendered as {half}/255; expected the sRGB encoding of "
        "0.5 (about 186), not the linear 128"
    )


def test_the_note_says_cancelled_rather_than_dim(out):
    note = out["readout"][f"{PI:.6f}"]["note"]
    assert "cancel" in note, f"the note at half a wavelength reads {note!r}"


def test_the_value_is_shown_in_wavelengths_and_radians(out):
    """A phase in radians alone means nothing to the reader this page is for."""
    value = out["readout"][f"{PI:.6f}"]["value"]
    assert "0.50" in value and "λ" in value, value
    assert "3.14" in value and "rad" in value, value


# --------------------------------------------------------------------- layout

def test_every_canvas_is_drawn_at_the_aspect_it_is_shown_at(out):
    """The errors.js plot bug, guarded before it can be repeated here."""
    for key, case in out["layout"].items():
        assert case["canvases"], f"{key}: the widget mounted no canvas at all"
        for c in case["canvases"]:
            shown_aspect = c["shownW"] / c["styleH"]
            bitmap_aspect = c["bitmapW"] / c["bitmapH"]
            assert abs(shown_aspect - bitmap_aspect) < 0.02, (
                f"{key}: canvas bitmap is {c['bitmapW']}x{c['bitmapH']} "
                f"(aspect {bitmap_aspect:.3f}) but is displayed "
                f"{c['shownW']}x{c['styleH']} (aspect {shown_aspect:.3f}); "
                "one axis is being stretched"
            )


def test_the_bitmap_follows_the_device_pixel_ratio(out):
    """A retina screen gets twice the bitmap, or the plot is drawn blurred."""
    one = out["layout"]["1042x1"]["canvases"][0]
    two = out["layout"]["1042x2"]["canvases"][0]
    assert two["bitmapW"] == 2 * one["bitmapW"]
    assert two["bitmapH"] == 2 * one["bitmapH"]
    assert two["styleH"] == one["styleH"], "the CSS height must not change with dpr"


def test_the_plot_is_capped_rather_than_letterboxed(out):
    """Capped on a desktop, full width on a phone."""
    assert out["plotCap"] == 640
    assert out["layout"]["1042x1"]["canvases"][0]["shownW"] == 640
    assert out["layout"]["300x1"]["canvases"][0]["shownW"] == 300


def test_the_widget_never_animates(source):
    """A rAF loop here could not be verified: the driven tab never fires one."""
    # The call, not the word: the file's own header explains why it has none.
    assert "requestAnimationFrame(" not in source, (
        "interfere.js must stay slider-driven; the hidden automation tab never "
        "fires animation frames, so an animated version cannot be checked"
    )


def test_the_stylesheet_carries_no_backtick(css):
    """One backtick anywhere in it terminates the template literal it lives in."""
    assert "`" not in css


def test_both_theme_paths_are_defined(css):
    """A widget that only styles one theme is unreadable in the other."""
    assert "prefers-color-scheme:dark" in css
    assert ':root[data-theme="dark"]' in css
    assert ':root[data-theme="light"]' in css
