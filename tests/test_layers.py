"""Tests for the differentiable torch layers (photonn/layers.py).

The layers must reproduce the NumPy physics exactly (in float64) and remain
autograd-friendly (finite gradients, |E| preserved by the phase mask).
"""
import numpy as np
import torch

from photonn.fields import Field
from photonn import propagate
from photonn.layers import AngularSpectrumLayer, PhaseMaskLayer

LAM = 532e-9


def test_angular_spectrum_layer_matches_numpy_reference():
    n, dx, z = 48, 8e-6, 4e-3
    rng = np.random.default_rng(1)
    data = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))

    ref = propagate.angular_spectrum(Field(data, dx, LAM), z).data
    layer = AngularSpectrumLayer(n, dx, LAM, z, dtype=torch.complex128)
    got = layer(torch.tensor(data, dtype=torch.complex128)[None])[0].numpy()

    assert np.allclose(got, ref, atol=1e-10)


def test_angular_spectrum_layer_batched_and_differentiable():
    n, dx, z = 32, 8e-6, 3e-3
    layer = AngularSpectrumLayer(n, dx, LAM, z)  # complex64
    x = torch.randn(3, n, n, dtype=torch.complex64, requires_grad=True)

    out = layer(x)
    assert out.shape == (3, n, n)

    (out.real**2 + out.imag**2).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad.abs()).all()


def test_phase_mask_layer_applies_exp_i_phi_and_preserves_amplitude():
    n = 16
    layer = PhaseMaskLayer(n)
    with torch.no_grad():
        layer.phi.copy_(torch.linspace(-3.0, 3.0, n * n).reshape(n, n))

    x = torch.randn(2, n, n, dtype=torch.complex64)
    out = layer(x)

    expected = x * torch.polar(torch.ones(n, n), layer.phi)
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.allclose(out.abs(), x.abs(), atol=1e-6)  # pure phase: |E| unchanged


def test_phase_mask_layer_default_init_is_transparent():
    n = 8
    layer = PhaseMaskLayer(n)  # init_std=0 -> zero phase -> identity
    x = torch.randn(1, n, n, dtype=torch.complex64)
    assert torch.allclose(layer(x), x, atol=1e-6)
