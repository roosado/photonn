"""Cross-check the browser angular-spectrum port against the NumPy reference.

The browser explorer (apps/web/asm.js) recomputes scalar diffraction live in the
JavaScript, so its physics must match photonn.propagate.angular_spectrum. This
test regenerates the reference intensities under Node (loading the same asm.js the
page ships) and asserts agreement to < 1e-6 on peak-normalized intensity, and that
the JS sampling-distance (z_crit) matches the Python check_sampling verdict.

Fixture: tests/fixtures/asm_reference.json, produced from photonn itself. If the
physics changes, regenerate it (see the generator noted in docs/wave_optics.md /
the plan). Requires Node on PATH; skips cleanly if absent.
"""
import json
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "asm_crosscheck_runner.js")
FIXTURE = os.path.join(HERE, "fixtures", "asm_reference.json")

TOL = 1e-6

node = shutil.which("node")


@pytest.mark.skipif(node is None, reason="node not on PATH; browser-physics cross-check skipped")
def test_asm_js_matches_numpy():
    assert os.path.exists(FIXTURE), "run the fixture generator first"
    proc = subprocess.run([node, RUNNER], capture_output=True, text=True)
    assert proc.returncode == 0, f"node runner failed:\n{proc.stderr}"

    results = json.loads(proc.stdout)
    assert results, "no cases in fixture"

    for r in results:
        # 1. the ported physics reproduces the NumPy field
        assert r["maxErr"] < TOL, (
            f"{r['name']} (n={r['n']}): max abs error {r['maxErr']:.2e} exceeds {TOL:.0e}"
        )
        # 2. the JS z_crit matches Python's, and the pass/fail verdict agrees
        assert abs(r["zCritJs"] - r["zCritRef"]) < 1e-12 * max(1.0, r["zCritRef"]), (
            f"{r['name']}: z_crit mismatch {r['zCritJs']} vs {r['zCritRef']}"
        )
        assert r["samplingOkJs"] == r["samplingOkRef"], (
            f"{r['name']}: sampling verdict disagrees (JS {r['samplingOkJs']} vs ref {r['samplingOkRef']})"
        )
