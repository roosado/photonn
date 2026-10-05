"""The activation widget: does it draw the device the activation page argues?

``apps/web/activation.js`` puts Williamson et al.'s (2020) electro-optic activation
in front of the reader in two modes. Its device mode draws the transmission
``T(P) = (1-a) cos^2((gP + phi_b)/2)`` and says in words what it does to weak
light; its network mode runs the trained two-layer chip. Both are held here to
something that is not the widget:

* **The curve is the closed form**, pointwise to 1e-12, at four devices.
* **field() is our own MZI.** The complex Eq. (6) the network mode applies equals
  ``sqrt(1-a) [B P(theta) B]_10 z`` built from ``photonn.mzi.beamsplitter`` and
  ``phase_shifter`` -- the same identity ``tests/test_eo_activation.py`` holds the
  library to -- and ``photonn.mzi.eo_activation`` itself, over 2 000 draws. Field,
  not just power: the self-phase is what the second mesh sees.
* **The endpoints by name**: fully open at ``P = pi/g`` when biased at pi, a
  low-power slope of exactly 2 there, and full transmission of weak light at 0.
* **The words carry the numbers**, and the widget refuses to invent a device.
* **Every canvas is drawn at the aspect it is shown at**, both modes, three widths,
  two pixel ratios -- the errors.js bug, guarded before it can be repeated.

The network mode's physics is held to the float64 reference separately, in
``tests/test_deep_mesh_web.py``.
"""
import json
import math
import os
import re
import shutil
import subprocess

import numpy as np
import pytest

from photonn import mzi

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "activation_runner.js")
WIDGET_JS = os.path.join(HERE, "..", "apps", "web", "activation.js")

node = shutil.which("node")


@pytest.fixture(scope="module")
def source():
    return open(WIDGET_JS, encoding="utf-8").read()


@pytest.fixture(scope="module")
def css(source):
    body = re.search(r"const CSS = `(.*?)`;", source, re.S)
    assert body, "could not find the CSS template literal in activation.js"
    return body.group(1)


@pytest.fixture(scope="module")
def out():
    if node is None:
        pytest.skip("node not on PATH")
    # utf-8 explicitly: the readouts print phi, mu and pi, and the Windows default
    # locale encoding turns them into mojibake the assertions then chase.
    proc = subprocess.run([node, RUNNER], capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, f"activation runner failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


# -------------------------------------------------------------------- physics

def test_the_drawn_curve_is_the_closed_form(out):
    for c in out["curves"]:
        p = np.array(c["p"])
        t = np.array(c["t"])
        closed = (1 - c["alpha"]) * np.cos(0.5 * (c["g"] * p + c["phiB"])) ** 2
        assert np.abs(t - closed).max() < 1e-12, c["phiB"]
        # Passivity, as the library asserts it: never above the tap's ceiling.
        assert t.max() <= 1 - c["alpha"] + 1e-15


def test_the_field_is_our_own_mzi(out):
    """sqrt(1-a)[B P(theta) B]_10 z at theta = -(phi_b + g|z|^2), and Eq. (6) itself."""
    b = mzi.beamsplitter(0.5)
    worst_mzi = worst_lib = 0.0
    for re_, im_, a, g, phi_b, (fr, fi) in out["field"]["draws"]:
        z = complex(re_, im_)
        f = complex(fr, fi)
        theta = -(phi_b + g * abs(z) ** 2)
        ref = math.sqrt(1 - a) * (b @ mzi.phase_shifter(theta) @ b)[1, 0] * z
        lib = mzi.eo_activation(z, alpha=a, g_phi=g, phi_b=phi_b)
        scale = max(abs(z), 1e-300)
        worst_mzi = max(worst_mzi, abs(f - ref) / scale)
        worst_lib = max(worst_lib, abs(f - complex(lib)) / scale)
    # Phases here reach ~37 rad; the bound is float64 round-off in cos() at that size.
    assert worst_mzi < 1e-13 and worst_lib < 1e-13, (worst_mzi, worst_lib)


def test_biased_at_pi_it_opens_fully_where_the_light_writes_pi(out):
    lm = out["landmarks"]
    assert abs(lm["tOpenPi"] - 0.9) < 1e-12
    assert abs(lm["pi"]["openP"] - math.pi / 0.3) < 1e-12
    assert abs(lm["pi"]["halfP"] - math.pi / 0.6) < 1e-12


def test_weak_light_goes_as_its_square_at_pi_bias(out):
    """The exact slope is 2x cot x with x = gP/2: still 1.99996 at P = 1e-2 pi/g."""
    assert abs(out["landmarks"]["slopePi"] - 2.0) < 1e-3
    assert out["landmarks"]["pi"]["slope"] == 2
    assert out["landmarks"]["pi"]["passesWeak"] is False


def test_weak_light_passes_whole_at_zero_bias(out):
    assert abs(out["landmarks"]["tWeakZero"] - 0.9) < 1e-12
    assert out["landmarks"]["zero"]["slope"] == 0
    assert out["landmarks"]["zero"]["passesWeak"] is True


# -------------------------------------------------------------------- readout

def _power(text):
    m = re.search(r"Opens fully at <b>([\d.]+) (\S+)</b>", text)
    return float(m.group(1)), m.group(2)


def test_the_readout_carries_the_numbers_it_claims(out):
    """The trained device opens at pi/g = 20 mW, and the note says so, from landmarks()."""
    r = out["readout"]["trained"]
    value, unit = _power(r["note"])
    scale = {"W": 1, "mW": 1e-3, "µW": 1e-6, "nW": 1e-9}[unit]
    assert abs(value * scale - r["landmarks"]["openP"]) / r["landmarks"]["openP"] < 0.03
    assert "blocks weak light" in r["note"] and "P²" in r["note"]
    assert "passes weak light" in out["readout"]["passing"]["note"]
    assert "P⁰" in out["readout"]["passing"]["note"]


def test_the_network_note_counts_photons_at_the_design_point(out):
    """At 1 mW and one 100 ps symbol, about one photon reaches the detectors."""
    note = out["readout"]["network"]["note"]
    m = re.search(r"leaves <b>([\d.]+) photons</b>", note)
    assert m, note
    assert 0.3 < float(m.group(1)) < 3.0


def test_the_bias_slider_can_sit_exactly_on_pi(out):
    """A 0.01-step range input turns pi into 3.14, 1.6 mrad off the dark setting.

    The browser snaps a range input's value to its step grid, and this stand-in does
    not, so the Node mount opened exactly on pi while the real page opened on 3.14 and
    told the reader the device passes "a fixed fraction, 0.000". Caught in a driven
    browser, held here.
    """
    assert out["readout"]["trained"]["steps"][0] == "any"


def test_a_bias_just_off_pi_is_described_as_nearly_dark(out):
    note = out["readout"]["nearDark"]["note"]
    assert "blocks weak light" in note and "sliver" in note and "P²" in note
    assert "0.000" not in note


def test_it_refuses_to_invent_a_device(out):
    assert out["errors"]["device"] and "no defaults" in out["errors"]["device"]


def test_the_symbols_survive_the_trip(source):
    """phi, mu and pi in the source, not mojibake (plan 11, lesson 3)."""
    assert "φ_b" in source and "µW" in source and "π" in source


# --------------------------------------------------------------------- layout

def test_every_canvas_is_drawn_at_the_aspect_it_is_shown_at(out):
    for key, cs in out["layout"].items():
        assert cs, f"{key}: no plot canvas mounted"
        for c in cs:
            shown = c["shownW"] / c["styleH"]
            bitmap = c["bitmapW"] / c["bitmapH"]
            assert abs(shown - bitmap) < 0.02, (key, c)


def test_the_bitmap_follows_the_device_pixel_ratio(out):
    one = out["layout"]["device:1042x1"][0]
    two = out["layout"]["device:1042x2"][0]
    assert two["bitmapW"] == 2 * one["bitmapW"] and two["bitmapH"] == 2 * one["bitmapH"]
    assert two["styleH"] == one["styleH"]


def test_the_plots_are_capped_rather_than_letterboxed(out):
    assert out["plotCap"] == 640
    assert out["layout"]["network:1042x1"][0]["shownW"] == 640
    assert out["layout"]["network:300x1"][0]["shownW"] == 300


def test_the_widget_never_animates(source):
    assert "requestAnimationFrame(" not in source


def test_the_stylesheet_carries_no_backtick(css):
    assert "`" not in css


def test_both_theme_paths_are_defined(css):
    assert "prefers-color-scheme:dark" in css
    assert ':root[data-theme="dark"]' in css
    assert ':root[data-theme="light"]' in css


def test_no_trained_constant_is_a_literal_in_the_widget(source):
    """The device comes from the bundle or the mount options, never from this file."""
    for literal in ("157.0", "0.05 * Math.PI", "eo_alpha = 0.1"):
        assert literal not in source, literal
