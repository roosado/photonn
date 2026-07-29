"""Analytic reference solutions and invariant checks.

Two roles:

1. **Analytic references** -- closed-form fields/patterns (Gaussian beam, Airy
   pattern, ...) that the physics functions are tested against. Every function in
   :mod:`photonn.propagate` and :mod:`photonn.mzi` has a corresponding analytic
   test built on these (CLAUDE.md convention).
2. **Invariant assertions** -- ``assert_*`` helpers used to enforce sampling,
   unitarity, and energy conservation at runtime, not only in tests.

Pure and NumPy-native.

References
----------
Gaussian beam: Siegman, *Lasers* (1986), ch. 17; Saleh & Teich, *Fundamentals of
Photonics*, ch. 3. Airy pattern: Goodman, *Introduction to Fourier Optics*,
sec. 4.4.2.
"""
from __future__ import annotations

import numpy as np
from scipy.special import j1

from photonn.fields import Field
from photonn.mzi import check_unitary
from photonn.propagate import check_sampling


def _centered_grid(n: int, dx: float):
    """Centred-origin spatial coordinate meshes (metres)."""
    x = (np.arange(n) - n // 2) * dx
    return np.meshgrid(x, x, indexing="xy")


# -- analytic reference solutions ---------------------------------------
def gaussian_beam(n: int, dx: float, wavelength: float, waist: float, z: float = 0.0) -> Field:
    """Return the analytic Gaussian-beam field at distance ``z`` from the waist.

    ``waist`` is the ``1/e`` amplitude radius ``w0`` at ``z = 0``. Uses the
    standard fundamental-mode expressions: Rayleigh range ``zR = pi w0^2 / lam``,
    spot size ``w(z)``, wavefront curvature ``R(z)``, and Gouy phase. The forward
    phase convention (``exp(+i(...))``) matches :func:`photonn.propagate.angular_spectrum`.
    """
    k = 2.0 * np.pi / wavelength
    w0 = waist
    zR = np.pi * w0**2 / wavelength

    x, y = _centered_grid(n, dx)
    r2 = x**2 + y**2

    wz = w0 * np.sqrt(1.0 + (z / zR) ** 2)
    amplitude = (w0 / wz) * np.exp(-r2 / wz**2)

    if z == 0.0:
        phase = np.zeros_like(r2)
    else:
        rz = z * (1.0 + (zR / z) ** 2)          # wavefront radius of curvature
        gouy = np.arctan(z / zR)
        phase = k * z + k * r2 / (2.0 * rz) - gouy

    return Field(amplitude * np.exp(1j * phase), dx, wavelength, z=z)


def airy_pattern(n: int, dx: float, wavelength: float, aperture_radius: float, z: float) -> Field:
    """Return the analytic Fraunhofer (Airy) pattern of a circular aperture.

    Amplitude ``2 J1(x)/x`` with ``x = k a r' / z`` (``a`` = ``aperture_radius``,
    ``r'`` the radial coordinate in the observation plane). Sampled on the same
    output grid as :func:`photonn.propagate.fraunhofer`
    (``dx' = lam * z / (n * dx)``) so the two can be compared pixel-for-pixel.
    """
    k = 2.0 * np.pi / wavelength
    dx_out = wavelength * z / (n * dx)
    x, y = _centered_grid(n, dx_out)
    r = np.sqrt(x**2 + y**2)

    arg = k * aperture_radius * r / z
    amplitude = np.ones_like(arg)              # limit of 2 J1(x)/x as x -> 0 is 1
    nz = arg != 0.0
    amplitude[nz] = 2.0 * j1(arg[nz]) / arg[nz]

    return Field(amplitude.astype(complex), dx_out, wavelength, z=z)


# -- runtime invariant assertions ---------------------------------------
def assert_sampling(field: Field, z: float, method: str = "angular_spectrum") -> None:
    """Raise ``ValueError`` if ``field`` is inadequately sampled to propagate ``z``."""
    report = check_sampling(field, z, method)
    if not report.ok:
        raise ValueError("Sampling criterion violated: " + " ".join(report.messages))


def assert_unitary(matrix, atol: float = 1e-10) -> None:
    """Raise ``ValueError`` if ``matrix`` is not unitary to tolerance ``atol``.

    Runtime wrapper of :func:`photonn.mzi.check_unitary`, mirroring
    :func:`assert_sampling`.
    """
    if not check_unitary(matrix, atol=atol):
        m = np.asarray(matrix)
        dev = float(np.max(np.abs(m.conj().T @ m - np.eye(m.shape[0]))))
        raise ValueError(f"matrix is not unitary: max|U*U - I| = {dev:.3e} > atol={atol:g}.")


def assert_energy_conserved(before: Field, after: Field, rtol: float = 1e-3) -> None:
    """Raise ``ValueError`` if total power changes by more than ``rtol`` across a lossless step."""
    p0 = before.power()
    p1 = after.power()
    if p0 == 0.0:
        raise ValueError("Reference field has zero power; cannot check conservation.")
    rel = abs(p1 - p0) / p0
    if rel > rtol:
        raise ValueError(
            f"Energy not conserved: relative power change {rel:.3e} exceeds rtol={rtol:.3e} "
            f"(before={p0:.6g}, after={p1:.6g})."
        )
