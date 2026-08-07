"""Cross-check the optics-sweep widget's geometry against the Python physics.

``apps/web/optics.js`` recomputes the diffractive reach from the closed form
``z*lambda/(2*dx^2)`` so its slider can move continuously between the measured
configurations. That makes it a second implementation of
:func:`photonn.propagate.diffraction_reach_px`, and this pins the two together --
the same guard ``tests/test_correspondence.py`` puts on the analogy figure.

It also checks the *generated data bundle* against the same physics, so a stale
``apps/web/optics_sweep.js`` (the committed copy that survives a fresh clone,
since ``exports/`` is gitignored) cannot quietly disagree with the code that
draws it.

Requires Node on PATH; skips cleanly if absent. Skips if the sweep has not been
run and exported yet.
"""
import json
import os
import shutil
import subprocess

import pytest

from photonn.propagate import diffraction_reach_px

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "optics_geom_runner.js")
BUNDLE = os.path.join(HERE, "..", "apps", "web", "optics_sweep.js")

CASES = [
    {"z_mm": 1.0, "dx": 8e-6, "wavelength": 532e-9},
    {"z_mm": 3.0, "dx": 8e-6, "wavelength": 532e-9},
    {"z_mm": 5.0, "dx": 8e-6, "wavelength": 532e-9},
    {"z_mm": 12.0, "dx": 8e-6, "wavelength": 532e-9},
    {"z_mm": 7.3, "dx": 8e-6, "wavelength": 532e-9},   # between measured points
]

node = shutil.which("node")
pytestmark = [
    pytest.mark.skipif(node is None, reason="node not on PATH; widget cross-check skipped"),
    pytest.mark.skipif(not os.path.exists(BUNDLE),
                       reason="apps/web/optics_sweep.js not generated yet "
                              "(run apps.sweep_optics then apps.sweep_report)"),
]


@pytest.fixture(scope="module")
def js():
    proc = subprocess.run([node, RUNNER, json.dumps(CASES)], capture_output=True, text=True)
    assert proc.returncode == 0, f"node runner failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def test_widget_reach_matches_python(js):
    """The live closed form must equal diffraction_reach_px, on and off the grid of measured z."""
    for case, got in zip(CASES, js["reach"]):
        # n only enters via the Matsushima band limit, inactive below z_crit;
        # the widget's form is the paraxial Nyquist reach, so compare at n=128.
        want = diffraction_reach_px(128, case["dx"], case["wavelength"], case["z_mm"] * 1e-3)
        assert got == pytest.approx(want, rel=1e-12), (
            f"z={case['z_mm']} mm: JS {got} vs Python {want}"
        )


def test_bundle_constants_match_the_operating_point(js):
    b = js["bundle"]
    assert b["grid"] == 128
    assert b["dx"] == pytest.approx(8e-6)
    assert b["wavelength"] == pytest.approx(532e-9)
    # z_crit = n*dx^2/lambda, quoted in mm.
    assert b["z_crit_mm"] == pytest.approx(128 * 8e-6**2 / 532e-9 * 1e3, rel=1e-9)


def test_bundle_reach_values_are_derived_not_typed(js):
    """Every stored reach must re-derive from the physics, per hop and in total."""
    b = js["bundle"]
    assert b["points"], "bundle carries no measured points"
    for p in b["points"]:
        want = diffraction_reach_px(b["grid"], b["dx"], b["wavelength"], p["z_mm"] * 1e-3)
        assert p["reach_hop"] == pytest.approx(want, rel=1e-9), f"z={p['z_mm']}mm per-hop reach"
        assert p["reach_total"] == pytest.approx(want * (p["layers"] + 1), rel=1e-9), (
            f"z={p['z_mm']}mm: total reach is not (n_layers + 1) hops"
        )


def test_bundle_accuracies_are_plausible(js):
    """Guard against a truncated or mis-parsed export reaching the site."""
    for p in js["bundle"]["points"]:
        assert 0.0 < p["acc"] <= 1.0, f"z={p['z_mm']}mm: accuracy {p['acc']} out of range"
        assert p["layers"] > 0, "the 0-mask floor belongs in floor_acc, not points"
