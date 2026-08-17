"""The mesh handoff must carry everything needed to rebuild the trained operator.

Schema 0.1.0 exported the MZI angles and nothing else, so the file was 108 parameters
short of the model (Sigma and the two output-phase screens) and the as-built side could
not reproduce the ideal accuracy. Reproducing it *exactly* is the anchor the whole error
budget rests on -- ``docs/tolerance_d2nn.md`` states it for the D2NN -- so this is the
Python-side twin of ``tests/test_d2nn_crosscheck.py``: rebuild the operator from the
handoff alone, in NumPy, and hold it to the torch model.

Two conventions this pins down, because both are easy to get backwards in MATLAB:

* The realised operator is ``U diag(sigma) V`` with **no conjugate transpose** anywhere.
  The docs call it ``U Sigma V^dagger``; since V is a free unitary the model class is the
  same, but a reimplementation must use V *as stored*.
* :class:`~photonn.layers.MZIMeshLayer` is row-vector (``x @ M.T``), the opposite of
  :func:`photonn.mzi.mesh_forward`'s column convention.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from photonn import mzi
from photonn.export import MESH_ORDER, MESH_TOPOLOGY, write_handoff
from photonn.layers import MZIMeshLayer
from photonn.models import MeshNetwork
from photonn.train import encode_modes

h5py = pytest.importorskip("h5py")

_REPO = Path(__file__).resolve().parent.parent
_HANDOFF = _REPO / "exports" / "mesh_phase3.h5"
_CKPT = _REPO / "exports" / "mesh_phase3.pt"


def mesh_matrix_from_settings(theta, phi, out_phase, n_modes):
    """Rebuild one Clements mesh operator from its stored angles, in NumPy.

    Mirrors :meth:`photonn.layers.MZIMeshLayer.matrix` using the pure-NumPy
    :func:`photonn.mzi.mzi_matrix`, so agreement is a check on the *stored numbers*
    rather than on torch running twice. This is the reference the MATLAB
    ``+meshmodel/mesh_matrix.m`` port is held to.
    """
    schedule = MZIMeshLayer(n_modes)._schedule
    m = np.eye(n_modes, dtype=complex)
    for column in schedule:
        layer = np.eye(n_modes, dtype=complex)
        for top, idx in column:
            layer[top:top + 2, top:top + 2] = mzi.mzi_matrix(theta[idx], phi[idx])
        m = layer @ m
    return np.diag(np.exp(1j * np.asarray(out_phase))) @ m


def operator_from_handoff(path):
    """Return ``(U diag(sigma) V, n_modes, n_classes, readout_gain)`` from the file alone."""
    with h5py.File(path, "r") as f:
        p = f["parameters"]
        n_modes = int(p.attrs["n_modes"])
        n_mzi = int(p.attrs["n_mzi_per_mesh"])
        theta, phi = p["phase_theta"][...], p["phase_phi"][...]
        sigma, out_phase = p["sigma"][...], p["out_phase"][...]
        op = f["operating_point"].attrs
        n_classes, gain = int(op["n_classes"]), float(op["readout_gain"])
        sigma_gain = float(op["sigma_gain"])

    v = mesh_matrix_from_settings(theta[:n_mzi], phi[:n_mzi], out_phase[0], n_modes)
    u = mesh_matrix_from_settings(theta[n_mzi:], phi[n_mzi:], out_phase[1], n_modes)
    return u @ np.diag(sigma.astype(complex)) @ v, n_modes, n_classes, gain, sigma_gain


def logits_from_operator(operator, x, n_classes, gain):
    """The readout: intensity on the first ``n_classes`` modes over total power."""
    out = x @ operator.T                          # row-vector convention
    intensity = np.abs(out) ** 2
    total = np.maximum(intensity.sum(axis=1, keepdims=True), 1e-12)
    return intensity[:, :n_classes] / total * gain


needs_exports = pytest.mark.skipif(
    not (_HANDOFF.exists() and _CKPT.exists()),
    reason="exports/mesh_phase3.{h5,pt} not present (gitignored; run apps.train_mesh)",
)


@needs_exports
def test_the_handoff_declares_its_topology_and_order():
    with h5py.File(_HANDOFF, "r") as f:
        p = f["parameters"]
        assert p.attrs["mesh_order"] == MESH_ORDER
        assert p.attrs["topology"] == MESH_TOPOLOGY
        n_modes, n_mzi = int(p.attrs["n_modes"]), int(p.attrs["n_mzi_per_mesh"])
    assert n_mzi == n_modes * (n_modes - 1) // 2
    assert MZIMeshLayer(n_modes).n_mzi == n_mzi


@needs_exports
def test_the_handoff_rebuilds_the_trained_operator():
    """Logits from the file alone must match the torch model's, not merely agree on argmax."""
    ckpt = torch.load(_CKPT, weights_only=False)
    saved = ckpt["args"]
    model = MeshNetwork(saved["modes"], saved["classes"], use_svd=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    operator, n_modes, n_classes, gain, sigma_gain = operator_from_handoff(_HANDOFF)
    assert n_modes == saved["modes"]

    with h5py.File(_HANDOFF, "r") as f:
        images = f["test_set/images"][...]
    x = images.reshape(len(images), -1).astype(np.complex128)   # row-major, as encoded

    ours = logits_from_operator(operator, x, n_classes, gain)
    with torch.no_grad():
        theirs = model(torch.as_tensor(x, dtype=torch.complex64)).numpy()

    # complex64 through 72 serial 36x36 products; 1e-5 is the float32 noise floor.
    np.testing.assert_allclose(ours, theirs, atol=1e-5)
    assert sigma_gain > 1.0            # the trained Sigma really did need normalising


@needs_exports
def test_the_handoff_reproduces_the_ideal_accuracy():
    """The number the MATLAB gate is held to. Recomputed, never hardcoded."""
    ckpt = torch.load(_CKPT, weights_only=False)
    saved = ckpt["args"]
    model = MeshNetwork(saved["modes"], saved["classes"], use_svd=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    operator, _, n_classes, gain, _ = operator_from_handoff(_HANDOFF)
    with h5py.File(_HANDOFF, "r") as f:
        images, labels = f["test_set/images"][...], f["test_set/labels"][...]
    x = images.reshape(len(images), -1).astype(np.complex128)

    from_file = (logits_from_operator(operator, x, n_classes, gain).argmax(1) == labels).mean()
    with torch.no_grad():
        from_model = (model(torch.as_tensor(x, dtype=torch.complex64)).numpy().argmax(1)
                      == labels).mean()
    assert from_file == pytest.approx(from_model, abs=5e-4)
    assert from_file > 0.7          # sanity: well above the 0.10 chance floor


@needs_exports
def test_the_stored_test_set_is_the_encoding_the_model_was_trained_on():
    """``test_set/images`` are already-normalised 6x6 amplitudes, not raw MNIST.

    MATLAB flattens them row-major into a 36-mode vector; if that ordering is wrong the
    accuracy comes out plausible but not equal, which is exactly the failure this catches.
    """
    from photonn.train import load_dataset

    ckpt = torch.load(_CKPT, weights_only=False)
    saved = ckpt["args"]
    test_ds = load_dataset("mnist", subset=saved["subset_test"], split="test")
    expected = encode_modes(torch.as_tensor(test_ds.images, dtype=torch.float32),
                            n_modes=saved["modes"]).abs().numpy()

    with h5py.File(_HANDOFF, "r") as f:
        images, labels = f["test_set/images"][...], f["test_set/labels"][...]
    np.testing.assert_allclose(images.reshape(len(images), -1), expected, atol=1e-6)
    np.testing.assert_array_equal(labels, test_ds.labels)


def test_passivize_leaves_the_logits_alone(rng):
    """Sign folds into the output phase, scale cancels in region/total. Exactly."""
    torch.manual_seed(20260817)
    model = MeshNetwork(8, 4, use_svd=True)
    with torch.no_grad():                       # signed, and larger than 1, as trained
        model.sigma.copy_(torch.as_tensor(rng.normal(0.0, 1.5, 8), dtype=torch.float32))
        model.v.out_phase.copy_(torch.as_tensor(rng.normal(0.0, 1.0, 8), dtype=torch.float32))

    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    sigma_p, out_phase_p, gain = mzi.passivize(sd["sigma"], sd["v.out_phase"])
    assert sigma_p.min() >= 0.0 and sigma_p.max() == pytest.approx(1.0)
    assert gain == pytest.approx(np.abs(sd["sigma"]).max())

    x = rng.normal(size=(16, 8)) + 1j * rng.normal(size=(16, 8))
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    def logits(sigma, out_phase_v):
        v = mesh_matrix_from_settings(sd["v.theta"], sd["v.phi"], out_phase_v, 8)
        u = mesh_matrix_from_settings(sd["u.theta"], sd["u.phi"], sd["u.out_phase"], 8)
        return logits_from_operator(u @ np.diag(sigma.astype(complex)) @ v, x, 4,
                                    model.readout_gain)

    np.testing.assert_allclose(logits(sigma_p, out_phase_p),
                               logits(sd["sigma"], sd["v.out_phase"]), atol=1e-12)


def test_a_roundtripped_payload_rebuilds_its_own_operator(tmp_path, mesh_payload):
    """The shape contract holds for any mesh, not just the one on disk."""
    path = tmp_path / "mesh.h5"
    write_handoff(path, **mesh_payload)
    with h5py.File(path, "r") as f:
        p = f["parameters"]
        n_modes, n_mzi = int(p.attrs["n_modes"]), int(p.attrs["n_mzi_per_mesh"])
        theta, out_phase = p["phase_theta"][...], p["out_phase"][...]
    assert theta.size == 2 * n_mzi and out_phase.shape == (2, n_modes)
    m = mesh_matrix_from_settings(theta[:n_mzi], mesh_payload["parameters"]["phase_phi"][:n_mzi],
                                  out_phase[0], n_modes)
    assert mzi.check_unitary(m)      # an ideal mesh is unitary before any error is applied
