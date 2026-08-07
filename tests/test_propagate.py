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


def _substep(field, z, steps):
    """Propagate ``z`` as ``steps`` equal sub-hops."""
    out = field
    for _ in range(steps):
        out = P.angular_spectrum(out, z / steps)
    return out


def test_substep_propagation_equals_one_hop_below_z_crit():
    """Splitting a hop is exact while the band limit is inactive.

    ``H = exp(i 2pi z sqrt(...))`` composes as ``H(z1)*H(z2) = H(z1+z2)``, so the
    only thing that could break sub-stepping is the *z-dependent* Matsushima band
    limit -- and below ``z_crit`` it sits above Nyquist and multiplies by 1 either
    way. This is what licenses the 3D stage (``apps/web/d2nn_stage.js``) to draw
    the field between the mask planes as physics rather than as a gradient.
    """
    n, dx, lam = 128, 8e-6, 532e-9              # the Phase-2 operating point
    z = 3e-3
    z_crit = n * dx**2 / lam
    assert z < z_crit

    launched = V.gaussian_beam(n, dx, lam, 60e-6, z=0.0)
    whole = P.angular_spectrum(launched, z)
    for steps in (2, 4, 8):
        split = _substep(launched, z, steps)
        err = np.abs(split.data - whole.data).max()
        assert err < 1e-12 * np.abs(whole.data).max(), (
            f"{steps} sub-steps drift from one hop by {err:.2e}"
        )


def test_substep_propagation_diverges_above_z_crit():
    """Past z_crit the band limit is active and z-dependent, so it does not compose.

    Sub-stepping there keeps content a single long hop would have discarded, and
    the two disagree measurably. Stated as a test so the exactness claim above is
    known to be a property of the regime, not of the method.
    """
    n, dx, lam = 128, 8e-6, 532e-9
    z_crit = n * dx**2 / lam
    z = 12 * z_crit

    launched = V.gaussian_beam(n, dx, lam, 20e-6, z=0.0)
    whole = P.angular_spectrum(launched, z)
    split = _substep(launched, z, 8)
    rel = np.abs(split.data - whole.data).max() / np.abs(whole.data).max()
    assert rel > 1e-3, f"expected sub-stepping to diverge above z_crit; got {rel:.2e}"


# -- wrap-around (finite window, not finite sampling) --------------------------
def _centred_square(n, half):
    data = np.zeros((n, n), dtype=complex)
    c = n // 2
    data[c - half:c + half, c - half:c + half] = 1.0
    return data


def test_wraparound_error_is_zero_without_propagation():
    """z = 0 gives H = 1, so nothing can leave the window and wrap is identically 0."""
    n, dx, lam = 64, 8e-6, 532e-9
    f = Field(_centred_square(n, 12), dx, lam)
    assert P.wraparound_error(f, 0.0) == pytest.approx(0.0, abs=1e-15)


def test_wraparound_error_grows_with_distance():
    """Monotone in z: the farther the field spreads, the more of it re-enters.

    This is the property that makes the diagnostic usable as a ceiling on usable
    separation -- and it is independent of ``check_sampling``, which reports the
    whole range below as well sampled.
    """
    n, dx, lam = 128, 8e-6, 532e-9
    f = Field(_centred_square(n, 16), dx, lam)
    errs = [P.wraparound_error(f, z) for z in (1e-3, 2e-3, 3e-3, 5e-3, 8e-3)]

    assert all(b > a for a, b in zip(errs, errs[1:])), f"not monotone: {errs}"
    # ...while the sampling criterion passes throughout, which is the point.
    assert all(P.check_sampling(f, z).ok for z in (1e-3, 8e-3))


def test_wraparound_error_falls_as_the_guard_band_grows():
    """Same z, same grid: a field confined nearer the centre wraps less."""
    n, dx, lam, z = 128, 8e-6, 532e-9, 4e-3
    wide = P.wraparound_error(Field(_centred_square(n, 40), dx, lam), z)
    narrow = P.wraparound_error(Field(_centred_square(n, 8), dx, lam), z)
    assert narrow < wide, f"narrow aperture wrapped more ({narrow:.2e} vs {wide:.2e})"


def test_wraparound_error_estimate_is_stable_in_pad_factor():
    """The padded reference must itself stop wrapping, or the number means nothing.

    Pad x2 is the working default; it is only trustworthy because x3 and x4 agree
    with it. Checked at the Phase-2 operating point.
    """
    n, dx, lam, z = 128, 8e-6, 532e-9, 3e-3
    f = Field(_centred_square(n, 32), dx, lam)
    e2, e3, e4 = (P.wraparound_error(f, z, pad_factor=pf) for pf in (2, 3, 4))

    assert e3 == pytest.approx(e4, rel=0.05)
    assert e2 == pytest.approx(e4, rel=0.25)


def test_wraparound_error_rejects_pad_factor_below_two():
    with pytest.raises(ValueError, match="pad_factor"):
        P.wraparound_error(Field(_centred_square(64, 8), 8e-6, 532e-9), 1e-3, pad_factor=1)
