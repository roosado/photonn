"""Tests for the Field container (photonn/fields.py).

The Field unit bookkeeping is implemented, so these run for real. Field behavior
that depends on propagation is left as a named placeholder for Phase 1.
"""
from __future__ import annotations

import numpy as np
import pytest

from photonn.fields import Field


def test_extent_and_grid():
    f = Field(np.zeros((16, 16)), dx=2e-6, wavelength=1.55e-6)
    assert f.n == 16
    assert f.extent == pytest.approx(16 * 2e-6)
    assert f.k == pytest.approx(2 * np.pi / 1.55e-6)


def test_real_input_promoted_to_complex():
    f = Field(np.ones((4, 4)), dx=1e-6, wavelength=1e-6)
    assert np.iscomplexobj(f.data)


def test_power_matches_manual_integral():
    data = np.full((8, 8), 2.0 + 0j)
    dx = 3e-6
    f = Field(data, dx=dx, wavelength=1e-6)
    expected = np.sum(np.abs(data) ** 2) * dx**2
    assert f.power() == pytest.approx(expected)


def test_coords_centered():
    f = Field(np.zeros((10, 10)), dx=1.0, wavelength=1e-6)
    coords = f.coords()
    assert coords.shape == (10,)
    assert coords[10 // 2] == pytest.approx(0.0)


def test_non_square_rejected():
    with pytest.raises(ValueError, match="square"):
        Field(np.zeros((4, 8)), dx=1e-6, wavelength=1e-6)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_nonpositive_dx_and_wavelength_rejected(bad):
    with pytest.raises(ValueError):
        Field(np.zeros((4, 4)), dx=bad, wavelength=1e-6)
    with pytest.raises(ValueError):
        Field(np.zeros((4, 4)), dx=1e-6, wavelength=bad)


def test_free_space_propagation_conserves_power():
    """A lossless propagation step must preserve Field.power() to tolerance."""
    from photonn import propagate as P
    from photonn import validate as V

    launched = V.gaussian_beam(256, 8e-6, 633e-9, 200e-6, z=0.0)
    propagated = P.angular_spectrum(launched, 0.02)
    assert propagated.power() == pytest.approx(launched.power(), rel=1e-4)
