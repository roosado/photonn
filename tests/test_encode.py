"""Tests for input encoding (photonn/train.py).

Only the pure encoding is exercised here; the MNIST download in ``load_dataset``
is intentionally not part of the unit suite (no network in tests).
"""
import pytest
import torch

from photonn.train import encode_input


def test_encode_schemes_shape_dtype_and_unit_norm():
    imgs = torch.rand(6, 28, 28)
    for scheme in ("amplitude", "phase", "both"):
        f = encode_input(imgs, scheme=scheme, n=64)
        assert f.shape == (6, 64, 64)
        assert f.dtype == torch.complex64
        l2 = f.abs().pow(2).sum(dim=(-2, -1))
        assert torch.allclose(l2, torch.ones(6), atol=1e-5)


def test_encode_amplitude_scheme_has_zero_phase():
    f = encode_input(torch.rand(3, 28, 28), scheme="amplitude", n=48)
    assert torch.allclose(f.imag, torch.zeros_like(f.imag), atol=1e-6)


def test_encode_phase_scheme_has_uniform_amplitude_in_window():
    f = encode_input(torch.rand(1, 28, 28), scheme="phase", n=48, input_frac=0.5)
    amp = f.abs()[0]
    illuminated = amp[amp > 0]
    # plane-wave through an aperture: every lit pixel shares one amplitude.
    assert torch.allclose(illuminated, illuminated.mean().expand_as(illuminated), atol=1e-5)


def test_encode_both_scheme_couples_amplitude_and_phase():
    # where the image is bright, both |E| and arg(E) should be non-trivial.
    img = torch.zeros(1, 28, 28)
    img[0, 10:18, 10:18] = 1.0
    f = encode_input(img, scheme="both", n=48, phase_scale=1.0)[0]
    lit = f.abs() > 0
    assert lit.any()
    assert f.angle()[lit].abs().max() > 0.0  # phase carries the image too


def test_encode_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        encode_input(torch.rand(1, 28, 28), scheme="hologram", n=32)
