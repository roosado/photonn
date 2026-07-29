"""Tests for the thin optical elements (photonn/elements.py)."""
import numpy as np
import pytest

from photonn.fields import Field
from photonn import elements as E

LAM = 633e-9


def _uniform_field(n=128, dx=1e-6):
    return Field(np.ones((n, n)), dx, LAM)


def test_circular_aperture_masks_outside_radius():
    n, dx, size = 128, 1e-6, 64e-6  # diameter -> radius 32 um
    out = E.aperture(_uniform_field(n, dx), "circular", size=size)

    x = out.coords()
    xx, yy = np.meshgrid(x, x, indexing="xy")
    inside = (xx**2 + yy**2) <= (size / 2.0) ** 2

    assert np.all(np.abs(out.data[inside]) == 1.0)
    assert np.all(out.data[~inside] == 0.0)
    # transmitted power equals the passed-sample count times the pixel area
    assert out.power() == pytest.approx(inside.sum() * dx**2)


def test_square_aperture_side_length():
    n, dx, size = 128, 1e-6, 40e-6
    out = E.aperture(_uniform_field(n, dx), "square", size=size)
    x = out.coords()
    xx, yy = np.meshgrid(x, x, indexing="xy")
    inside = (np.abs(xx) <= size / 2.0) & (np.abs(yy) <= size / 2.0)
    assert np.all(out.data[~inside] == 0.0)
    assert np.all(np.abs(out.data[inside]) == 1.0)


def test_aperture_requires_size_and_known_shape():
    with pytest.raises(ValueError, match="size"):
        E.aperture(_uniform_field(), "circular", size=None)
    with pytest.raises(ValueError, match="shape"):
        E.aperture(_uniform_field(), "triangular", size=1e-5)


def test_thin_lens_is_pure_phase_with_zero_center_phase():
    n, dx, f = 128, 2e-6, 0.05
    out = E.thin_lens(_uniform_field(n, dx), focal_length=f)

    # Pure phase: amplitude unchanged everywhere.
    assert np.allclose(np.abs(out.data), 1.0)

    # Centre (r=0) picks up no phase; a known off-axis point matches -k r^2 / (2f).
    c = n // 2
    assert out.phase()[c, c] == pytest.approx(0.0, abs=1e-12)

    x = out.coords()
    r2 = x[c + 10] ** 2  # on-axis row, 10 pixels off centre in x
    expected = -out.k * r2 / (2.0 * f)
    got = np.angle(out.data[c, c + 10])
    assert np.exp(1j * got) == pytest.approx(np.exp(1j * expected))


def test_phase_mask_is_pure_phase():
    n, dx = 32, 1e-6
    rng = np.random.default_rng(0)
    f = _uniform_field(n, dx)
    phase = rng.standard_normal((n, n))
    out = E.phase_mask(f, phase)

    assert np.allclose(np.abs(out.data), np.abs(f.data))          # |E| unchanged
    assert np.allclose(out.phase(), np.angle(np.exp(1j * phase)))  # arg(E) = wrapped phase


def test_amplitude_mask_scales_power_by_transmission_squared():
    n, dx = 32, 1e-6
    f = _uniform_field(n, dx)
    out = E.amplitude_mask(f, np.full((n, n), 0.5))
    assert out.power() == pytest.approx(0.25 * f.power())


def test_masks_reject_shape_mismatch_and_out_of_range():
    f = _uniform_field(16)
    with pytest.raises(ValueError, match="grid"):
        E.phase_mask(f, np.zeros((8, 8)))
    with pytest.raises(ValueError, match="grid"):
        E.amplitude_mask(f, np.zeros((8, 8)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        E.amplitude_mask(f, np.full((16, 16), 2.0))
