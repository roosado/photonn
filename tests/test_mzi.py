"""Analytic tests for the MZI mesh (photonn/mzi.py) and the torch mesh layer.

The decompositions are verified by reconstruction (the correctness anchor), and
the torch mesh is checked against a NumPy replication.
"""
import numpy as np
import pytest
import torch
from scipy.stats import unitary_group

from photonn import mzi
from photonn.layers import MZIMeshLayer
from photonn.models import MeshNetwork
from photonn.train import encode_modes
from photonn import validate


def test_mzi_matrix_is_unitary():
    for theta in np.linspace(0.0, np.pi, 9):
        for phi in np.linspace(0.0, 2 * np.pi, 9):
            assert mzi.check_unitary(mzi.mzi_matrix(theta, phi))


def test_beamsplitter_is_unitary_with_correct_split():
    b = mzi.beamsplitter(0.5)
    assert mzi.check_unitary(b)
    assert abs(b[0, 1]) ** 2 == pytest.approx(0.5)          # 50:50 cross power
    assert abs(mzi.beamsplitter(0.3)[0, 1]) ** 2 == pytest.approx(0.3)


@pytest.mark.parametrize("n", [2, 3, 6, 8])
def test_clements_reconstructs_unitary(n):
    u = unitary_group.rvs(n, random_state=np.random.default_rng(n))
    settings = mzi.clements_decompose(u)
    assert len(settings["ops"]) == n * (n - 1) // 2
    assert np.allclose(mzi.reconstruct(settings), u, atol=1e-10)


@pytest.mark.parametrize("n", [2, 3, 6, 8])
def test_reck_reconstructs_unitary(n):
    u = unitary_group.rvs(n, random_state=np.random.default_rng(100 + n))
    settings = mzi.reck_decompose(u)
    assert len(settings["ops"]) == n * (n - 1) // 2
    assert np.allclose(mzi.reconstruct(settings), u, atol=1e-10)


def test_svd_layer_represents_real_matrix():
    rng = np.random.default_rng(7)
    m = rng.standard_normal((6, 6))
    sv = mzi.svd_decompose(m)
    assert np.allclose(mzi.svd_reconstruct(sv), m, atol=1e-10)


def test_mesh_forward_matches_matrix_multiply():
    rng = np.random.default_rng(3)
    u = unitary_group.rvs(5, random_state=rng)
    settings = mzi.clements_decompose(u)
    x = rng.standard_normal(5) + 1j * rng.standard_normal(5)
    assert np.allclose(mzi.mesh_forward(settings, x), u @ x, atol=1e-10)


def test_assert_unitary_raises_for_non_unitary():
    validate.assert_unitary(unitary_group.rvs(4, random_state=np.random.default_rng(1)))
    with pytest.raises(ValueError, match="not unitary"):
        validate.assert_unitary(np.ones((4, 4)))


def test_mzi_mesh_layer_matches_numpy_and_is_unitary():
    torch.manual_seed(0)
    n = 6
    layer = MZIMeshLayer(n)
    m = layer.matrix().detach().numpy()
    assert layer.n_mzi == n * (n - 1) // 2
    assert mzi.check_unitary(m, atol=1e-4)                   # complex64 precision

    th, ph, op = (layer.theta.detach().numpy(), layer.phi.detach().numpy(),
                  layer.out_phase.detach().numpy())
    ref = np.eye(n, dtype=complex)
    for pairs in layer._schedule:
        block = np.eye(n, dtype=complex)
        for mode, idx in pairs:
            block[mode:mode + 2, mode:mode + 2] = mzi.mzi_matrix(th[idx], ph[idx])
        ref = block @ ref
    ref = np.diag(np.exp(1j * op)) @ ref
    assert np.allclose(m, ref, atol=1e-5)


def test_mesh_network_forward_and_gradients():
    torch.manual_seed(0)
    net = MeshNetwork(36, 10)
    x = encode_modes(torch.rand(8, 28, 28), n_modes=36)
    logits = net(x)
    assert logits.shape == (8, 10)

    torch.nn.functional.cross_entropy(logits, torch.arange(8) % 10).backward()
    assert torch.isfinite(net.u.theta.grad).all()
    assert torch.isfinite(net.sigma.grad).all()
