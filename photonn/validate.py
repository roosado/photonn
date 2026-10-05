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

import contextlib

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


def eo_activation_reference(z, *, alpha: float, g_phi: float, phi_b: float) -> np.ndarray:
    """The electro-optic activation rebuilt from the MZI primitives, one mode at a time.

    Williamson et al. (2020) describe the device, not an equation to trust: a tap of
    ``alpha``, then the remaining ``sqrt(1-alpha)`` of the field crossing an MZI whose
    internal phase the tapped light set. So build that MZI from
    :func:`photonn.mzi.beamsplitter` and :func:`photonn.mzi.phase_shifter` at
    ``theta = -(phi_b + g|z|^2)`` and read the cross port,
    ``sqrt(1-alpha) [B P(theta) B]_{10} z``. :func:`photonn.mzi.eo_activation`
    (the paper's closed form, Eq. 6) must equal this to round-off -- the
    activation's counterpart of Clements reconstruction.

    Deliberately a Python loop over 2x2 products: it is the slow, obvious
    construction the fast one is checked against.
    """
    from photonn.mzi import beamsplitter, phase_shifter

    z = np.asarray(z, dtype=complex)
    b = beamsplitter(0.5)
    out = np.empty_like(z)
    for idx, zi in np.ndenumerate(z):
        theta = -(phi_b + g_phi * abs(zi) ** 2)
        out[idx] = np.sqrt(1.0 - alpha) * (b @ phase_shifter(theta) @ b)[1, 0] * zi
    return out


# -- runtime invariant assertions ---------------------------------------
#: Whether the propagators enforce the sampling criterion as they run.
#:
#: CLAUDE.md asks for "sampling and unitarity checks as runtime assertions, not
#: just tests", and this module's own docstring repeated the claim -- while the
#: three ``assert_*`` functions below had no caller anywhere outside two test
#: lines. They were a stated rule that nothing held: ``angular_spectrum`` never
#: checked its sampling, so a mis-sampled propagation returned a quietly aliased
#: field that some later figure inherited without complaint.
#:
#: On by default, because that is what the rule means. Turn it off around a call
#: that is *deliberately* outside the criterion -- showing a reader what aliasing
#: looks like, or measuring how far past z_crit a design can be pushed -- with
#: :func:`relaxed`, which says so at the call site instead of silently.
STRICT = True


@contextlib.contextmanager
def relaxed():
    """Suspend :data:`STRICT` enforcement for a block, and say so out loud.

    >>> with validate.relaxed():            # doctest: +SKIP
    ...     aliased = angular_spectrum(field, 4 * z_crit)
    """
    global STRICT
    previous = STRICT
    STRICT = False
    try:
        yield
    finally:
        STRICT = previous


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
