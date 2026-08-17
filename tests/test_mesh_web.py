"""The chip widget must draw the chip that was trained, not a plausible one.

``/tolerance`` closes on the chip's own error budget, and carries one widget for
it: coupler imbalance on the MZI mesh, the source that has no meaning for a phase
mask and that ``docs/tolerance_mesh.md`` measures binding level with phase error.
It renders

    |O| = |U . diag(sigma) . V|

from the trained settings, quantised into ``apps/web/mesh_weights.js`` by
``apps.export_mesh_web``. Two things can go wrong quietly and neither shows up as
a broken page:

* **the bundle drifts from the model** -- it is committed, because ``exports/*.h5``
  is gitignored and a fresh clone still has to build the site, so nothing forces
  it to be regenerated when the mesh is;
* **the JavaScript rebuild is wrong** -- a transposed index, the closed-form 2x2
  block where the factored one is needed, a Clements schedule starting on the
  wrong parity. Any of those draws a confident picture of a different chip.

So the committed codes are decoded here and rebuilt in NumPy through
``photonn.mzi``, the shipped JS is run under Node and its operator compared
against that, and -- when the gitignored handoff is present -- the codes are
checked against the numbers they were quantised from.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

# The NumPy reference for one Clements mesh lives with the handoff-sufficiency
# test, and is imported rather than copied so both suites hold the JS and the
# MATLAB port to the *same* reference. pytest's prepend import mode puts this
# directory on sys.path, so the bare module name resolves.
from test_mesh_handoff_is_sufficient import mesh_matrix_from_settings

from apps.export_mesh_web import LEVELS, TWO_PI, decode

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RUNNER = os.path.join(HERE, "mesh_operator_runner.js")
WEIGHTS_JS = os.path.join(REPO, "apps", "web", "mesh_weights.js")
HANDOFF = os.path.join(REPO, "exports", "mesh_phase3.h5")

#: One 16-bit code is 2*pi/65536 = 9.6e-5 rad, and 72 columns of them compound.
#: Measured agreement is 4.5e-5 on a peak entry of 0.345; this leaves room.
OPERATOR_TOL = 5e-4

node = shutil.which("node")


@pytest.fixture(scope="module")
def bundle():
    """The committed bundle, as a dict, without going through Node."""
    text = open(WEIGHTS_JS, encoding="utf-8").read()
    start = text.index("var W = ") + len("var W = ")
    end = text.index(";\n  if (typeof module", start)
    return json.loads(text[start:end])


@pytest.fixture(scope="module")
def js():
    if node is None:
        pytest.skip("node not on PATH")
    proc = subprocess.run([node, RUNNER], capture_output=True, text=True)
    assert proc.returncode == 0, f"mesh operator runner failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def _operator(theta, phi, out_phase, n, n_mzi):
    """``U . diag(sigma) . V`` is assembled by the caller; this is one mesh pair."""
    v = mesh_matrix_from_settings(theta[:n_mzi], phi[:n_mzi], out_phase[:n], n)
    u = mesh_matrix_from_settings(theta[n_mzi:], phi[n_mzi:], out_phase[n:], n)
    return u, v


# ------------------------------------------------------------------- the bundle

def test_the_bundle_is_shaped_like_the_trained_mesh(bundle):
    n, n_mzi = bundle["n"], bundle["n_mzi"]
    assert (n, n_mzi) == (36, 630), "the trained Phase-3 mesh is 36 modes, 630 MZIs"
    assert n_mzi == n * (n - 1) // 2, "a Clements rectangle has N(N-1)/2 MZIs"
    assert bundle["bits"] == 16
    assert len(decode(bundle["theta_b64"], TWO_PI)) == 2 * n_mzi
    assert len(decode(bundle["phi_b64"], TWO_PI)) == 2 * n_mzi
    assert len(decode(bundle["sigma_b64"], 1.0)) == n
    assert len(decode(bundle["out_phase_b64"], TWO_PI)) == 2 * n


def test_the_bundle_carries_a_passive_mesh(bundle):
    """A chip that amplifies is not a chip. ``photonn.mzi.passivize`` guarantees
    this upstream; if it ever regressed, the widget would be drawing free gain."""
    sigma = decode(bundle["sigma_b64"], 1.0)
    assert sigma.min() >= 0.0
    assert sigma.max() <= 1.0


def test_the_two_meshes_are_unitary_and_sigma_is_the_only_loss(bundle):
    """``||U diag(s) V||_F = ||s||_2`` exactly when U and V are unitary.

    One line that tests both meshes' unitarity *and* that sigma is applied
    between them rather than anywhere else -- if either mesh were mis-assembled
    the Frobenius norm would not survive.
    """
    n, n_mzi = bundle["n"], bundle["n_mzi"]
    theta = decode(bundle["theta_b64"], TWO_PI)
    phi = decode(bundle["phi_b64"], TWO_PI)
    sigma = decode(bundle["sigma_b64"], 1.0)
    out_phase = decode(bundle["out_phase_b64"], TWO_PI)
    u, v = _operator(theta, phi, out_phase, n, n_mzi)
    for name, m in (("U", u), ("V", v)):
        assert np.allclose(m.conj().T @ m, np.eye(n), atol=1e-10), f"{name} is not unitary"
    operator = u @ np.diag(sigma.astype(complex)) @ v
    assert np.linalg.norm(operator) == pytest.approx(np.linalg.norm(sigma), rel=1e-10)


# ------------------------------------------------------------- the shipped JS

def test_the_widget_rebuilds_the_operator_numpy_builds(bundle, js):
    """The check this file exists for: JS and NumPy, same settings, same matrix."""
    n, n_mzi = bundle["n"], bundle["n_mzi"]
    theta = decode(bundle["theta_b64"], TWO_PI)
    phi = decode(bundle["phi_b64"], TWO_PI)
    sigma = decode(bundle["sigma_b64"], 1.0)
    out_phase = decode(bundle["out_phase_b64"], TWO_PI)
    # The JS decode must agree first, or a later mismatch is ambiguous.
    assert np.allclose(js["theta"], theta, atol=1e-12)
    assert np.allclose(js["phi"], phi, atol=1e-12)
    assert np.allclose(js["sigma"], sigma, atol=1e-12)
    assert np.allclose(js["outPhase"], out_phase, atol=1e-12)

    u, v = _operator(theta, phi, out_phase, n, n_mzi)
    reference = np.abs(u @ np.diag(sigma.astype(complex)) @ v)
    got = np.array(js["ideal"]).reshape(n, n)
    assert got.shape == reference.shape
    assert np.max(np.abs(got - reference)) < OPERATOR_TOL, (
        "the widget's operator is not the one NumPy builds from the same numbers"
    )


def test_two_ideal_couplers_are_the_same_as_no_couplers(js):
    """The ideal panel and a zero-imbalance draw take different code paths.

    The ideal panel skips the split array entirely; a zero draw runs the factored
    block with both splits at 0.5. If those disagree, the panel labelled "as
    designed" is not what the slider returns to at zero.
    """
    a = np.array(js["ideal"])
    b = np.array(js["zeroEps"])
    assert np.max(np.abs(a - b)) < 1e-12


def test_imbalance_moves_the_whole_operator_not_just_the_couplers(js):
    """The claim the widget makes in words, asserted on its own numbers.

    72 columns in series means a mis-split coupler in column 3 is still visible at
    the output, so the disturbance has to be spread across the matrix rather than
    concentrated. Fewer than half the entries moving would make the page's
    serial-accumulation sentence false.
    """
    n = js["n"]
    ideal = np.array(js["ideal"]).reshape(n, n)
    stressed = np.array(js["stressed"]).reshape(n, n)
    diff = np.abs(stressed - ideal)
    rms = np.sqrt((diff ** 2).mean()) / np.sqrt((ideal ** 2).mean())
    assert 0.02 < rms < 0.30, f"a 0.01 split error moved the operator by {rms:.1%}"
    moved = (diff > 0.05 * diff.max()).mean()
    assert moved > 0.5, f"only {moved:.0%} of the operator moved; that is not serial"


# --------------------------------------------------------- against the handoff

@pytest.mark.skipif(not os.path.exists(HANDOFF),
                    reason="exports/mesh_phase3.h5 not present (gitignored)")
def test_the_committed_bundle_has_not_drifted_from_the_handoff(bundle):
    """Committed generated files go stale silently; this is the alarm."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(HANDOFF, "r") as f:
        p = f["parameters"]
        theta, phi = p["phase_theta"][...], p["phase_phi"][...]
        sigma, out_phase = p["sigma"][...], p["out_phase"][...]

    step = TWO_PI / LEVELS
    for name, span, stored, coded in (
        ("theta", TWO_PI, np.mod(theta, TWO_PI), decode(bundle["theta_b64"], TWO_PI)),
        ("phi", TWO_PI, np.mod(phi, TWO_PI), decode(bundle["phi_b64"], TWO_PI)),
        ("out_phase", TWO_PI, np.mod(out_phase.reshape(-1), TWO_PI),
         decode(bundle["out_phase_b64"], TWO_PI)),
    ):
        assert np.max(np.abs(coded - stored)) <= step, (
            f"{name} in mesh_weights.js is more than one code away from the "
            "handoff -- re-run `python -m apps.export_mesh_web`"
        )
    assert np.max(np.abs(decode(bundle["sigma_b64"], 1.0) - sigma)) <= 1.0 / LEVELS
