"""Analytic tests for the diffraction propagators (photonn/propagate.py).

Each propagator is checked against a closed-form result from photonn.validate.
Tolerances are generous relative to the observed error (which is near machine
precision for the Gaussian and Fresnel cases).
"""
import numpy as np
import pytest

from photonn.fields import Field
from photonn import propagate as P
from photonn import validate as V
from photonn import elements as E

LAM = 633e-9  # HeNe wavelength (m)


def _norm(field):
    intensity = field.intensity()
    return intensity / intensity.max()


def _second_moment_width(field):
    """1/e amplitude radius via the intensity second moment."""
    intensity = field.intensity()
    x = field.coords()
    xx, yy = np.meshgrid(x, x, indexing="xy")
    r2 = xx**2 + yy**2
    return np.sqrt(2.0 * np.sum(intensity * r2) / np.sum(intensity))


def test_angular_spectrum_matches_gaussian_beam():
    """angular_spectrum of a Gaussian waist matches validate.gaussian_beam at z."""
    n, dx, w0, z = 256, 8e-6, 200e-6, 0.1  # z >> z_crit; band-limited ASM still exact here
    launched = V.gaussian_beam(n, dx, LAM, w0, z=0.0)
    numeric = P.angular_spectrum(launched, z)
    analytic = V.gaussian_beam(n, dx, LAM, w0, z=z)

    rms = np.sqrt(np.mean((_norm(numeric) - _norm(analytic)) ** 2))
    assert rms < 1e-4

    zR = np.pi * w0**2 / LAM
    w_expected = w0 * np.sqrt(1.0 + (z / zR) ** 2)
    assert _second_moment_width(numeric) == pytest.approx(w_expected, rel=5e-3)


def test_fraunhofer_circular_aperture_is_airy():
    """fraunhofer of a circular aperture matches validate.airy_pattern."""
    n, dx, a, z = 1024, 2e-6, 100e-6, 0.2  # z well beyond the aperture far-field distance
    aperture = E.aperture(Field(np.ones((n, n)), dx, LAM), "circular", size=2 * a)
    far = P.fraunhofer(aperture, z)
    airy = V.airy_pattern(n, dx, LAM, a, z)

    # Same output grid, so compare pixel-for-pixel over the bright central region.
    numeric, analytic = _norm(far), _norm(airy)
    mask = analytic > 1e-2
    assert np.allclose(numeric[mask], analytic[mask], atol=5e-3)

    # First Airy null at 1.22 * lam * z / D lands on the expected ring.
    first_null_radius = 1.22 * LAM * z / (2 * a)
    assert far.dx == pytest.approx(LAM * z / (n * dx))
    assert first_null_radius / far.dx > 5  # resolved by several pixels


def test_fresnel_agrees_with_angular_spectrum_in_validity_range():
    """fresnel and angular_spectrum agree where the paraxial approximation holds."""
    n, dx, w0, z = 256, 8e-6, 200e-6, 0.02  # paraxial and z <= z_crit
    launched = V.gaussian_beam(n, dx, LAM, w0, z=0.0)
    fres = P.fresnel(launched, z)
    asm = P.angular_spectrum(launched, z)
    assert np.max(np.abs(_norm(fres) - _norm(asm))) < 1e-3


def test_propagation_conserves_energy():
    """Total power is preserved across a lossless angular-spectrum step."""
    n, dx, w0, z = 256, 8e-6, 200e-6, 0.02
    launched = V.gaussian_beam(n, dx, LAM, w0, z=0.0)
    propagated = P.angular_spectrum(launched, z)
    V.assert_energy_conserved(launched, propagated, rtol=1e-4)  # raises on failure


def test_check_sampling_flags_aliasing():
    """check_sampling returns ok=False when the grid under-samples the field."""
    n, dx = 256, 8e-6
    field = Field(np.ones((n, n)), dx, LAM)
    z_crit = n * dx**2 / LAM

    assert P.check_sampling(field, 0.1 * z_crit, "angular_spectrum").ok is True
    report = P.check_sampling(field, 10 * z_crit, "angular_spectrum")
    assert report.ok is False
    assert report.messages  # explains the violation

    # Fraunhofer: invalid in the near field, valid far away.
    assert P.check_sampling(field, 0.01, "fraunhofer").ok is False
    assert P.check_sampling(field, 1e3, "fraunhofer").ok is True


def test_check_sampling_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        P.check_sampling(Field(np.ones((8, 8)), 1e-6, LAM), 0.1, "nonsense")
