"""Free-space scalar diffraction propagators.

The angular-spectrum method is the reference propagator; Fresnel and Fraunhofer
are its paraxial near-field and far-field approximations, each valid only within
a stated range. Every propagator here gets a corresponding analytic test in
:mod:`photonn.validate`, and sampling adequacy is checked at runtime (not only in
tests) -- both are CLAUDE.md working conventions.

Scalar diffraction only: no vector or polarization effects (scope boundary).

Functions are pure and NumPy-native. The differentiable equivalents live in
:mod:`photonn.layers`; do not import autodiff machinery here.

References
----------
Goodman, *Introduction to Fourier Optics*, 3rd ed., chs. 3-5 (angular spectrum,
Fresnel, Fraunhofer). Voelz, *Computational Fourier Optics* (SPIE, 2011),
ch. 5 (sampling / critical distance). Matsushima & Shimobaba, "Band-Limited
Angular Spectrum Method...", IEEE Trans. Image Process. 18(11):2646 (2009)
(band limit that keeps the angular spectrum alias-free in the far field).
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field

import numpy as np

from photonn.fields import Field


@dataclass
class SamplingReport:
    """Outcome of a sampling-adequacy check for one propagation step.

    Attributes
    ----------
    ok : bool
        True if the grid adequately samples the field after propagation.
    method : str
        Propagator the check applies to.
    messages : list[str]
        Human-readable notes on any violated criteria. Surfaced live in the
        diffraction explorer (Phase 1 deliverable) and used to build runtime
        assertion messages.
    """

    ok: bool
    method: str
    messages: list = _dc_field(default_factory=list)


def _freq_grid(n: int, dx: float):
    """Return centred-origin spatial-frequency meshes (cycles/m) for an ``n`` x ``n`` grid.

    Frequencies are in ``np.fft.fftfreq`` (unshifted) order to match ``fft2`` of an
    ``ifftshift``-ed field.
    """
    f = np.fft.fftfreq(n, d=dx)
    return np.meshgrid(f, f, indexing="xy")


def angular_spectrum_transfer(n: int, dx: float, wavelength: float, z: float) -> np.ndarray:
    """Band-limited angular-spectrum transfer function ``H`` for one propagation step.

    Returns the complex ``(n, n)`` multiplier applied to the field spectrum, in
    ``np.fft.fftfreq`` (unshifted) order so it multiplies ``fft2(ifftshift(E))``
    directly. Depends only on the grid geometry ``(n, dx, wavelength)`` and the
    distance ``z`` -- none of which are trainable -- so the differentiable
    :class:`photonn.layers.AngularSpectrumLayer` precomputes it here once and
    reuses it as a constant buffer. This is the single source of truth for the
    ASM physics shared by the NumPy propagator and the torch layer.

    ``H = exp(i 2pi z sqrt(1/lam^2 - fx^2 - fy^2))``; the complex square root
    gives evanescent decay where ``fx^2 + fy^2 > 1/lam^2`` (for ``z > 0``). The
    transfer function is then band-limited per Matsushima & Shimobaba (2009): the
    local frequency of ``H`` reaches the sampling limit at ``u_limit``, and
    everything beyond it is zeroed to prevent aliasing into the far field.
    """
    fx, fy = _freq_grid(n, dx)

    kz_over_2pi = np.sqrt((1.0 / wavelength) ** 2 - fx**2 - fy**2 + 0j)
    H = np.exp(1j * 2.0 * np.pi * z * kz_over_2pi)

    du = 1.0 / (n * dx)
    u_limit = 1.0 / (wavelength * np.sqrt((2.0 * du * z) ** 2 + 1.0))
    H *= (np.abs(fx) <= u_limit) & (np.abs(fy) <= u_limit)
    return H


def angular_spectrum(field: Field, z: float) -> Field:
    """Propagate ``field`` a distance ``z`` by the angular-spectrum method.

    Exact for scalar, monochromatic fields up to the band limit of the grid --
    no paraxial approximation. Evanescent components decay; the transfer function
    is band-limited per Matsushima & Shimobaba (2009) so the method stays
    alias-free into the far field (at the cost of discarding content beyond the
    propagating band). Reference for :func:`fresnel`, :func:`fraunhofer`, and the
    torch layer in :mod:`photonn.layers`.
    """
    n, dx, lam = field.n, field.dx, field.wavelength
    H = angular_spectrum_transfer(n, dx, lam, z)

    spectrum = np.fft.fft2(np.fft.ifftshift(field.data))
    out = np.fft.fftshift(np.fft.ifft2(spectrum * H))
    return Field(out, dx, lam, z=field.z + z, units=field.units)


def fresnel(field: Field, z: float) -> Field:
    """Fresnel (paraxial near-field) approximation of :func:`angular_spectrum`.

    Transfer-function form on the *same* grid, so it is directly comparable to
    :func:`angular_spectrum` pixel-for-pixel. Valid where the paraxial
    approximation holds (small angles) and the transfer function is well sampled
    (``|z| <= z_crit``; see :func:`check_sampling`).
    """
    n, dx, lam = field.n, field.dx, field.wavelength
    fx, fy = _freq_grid(n, dx)

    # Paraxial expansion of the ASM transfer function.
    H = np.exp(1j * field.k * z) * np.exp(-1j * np.pi * lam * z * (fx**2 + fy**2))

    spectrum = np.fft.fft2(np.fft.ifftshift(field.data))
    out = np.fft.fftshift(np.fft.ifft2(spectrum * H))
    return Field(out, dx, lam, z=field.z + z, units=field.units)


def fraunhofer(field: Field, z: float) -> Field:
    """Fraunhofer (far-field) approximation of :func:`angular_spectrum`.

    Single-FFT far-field form. The output plane is **resampled**: the returned
    field has spacing ``dx' = lam * z / (n * dx)``. Valid beyond the Fraunhofer
    distance ``z >= 2 D^2 / lam`` (``D`` = source extent; see
    :func:`check_sampling`).
    """
    n, dx, lam = field.n, field.dx, field.wavelength
    k = field.k

    dx_out = lam * z / (n * dx)
    x_out = (np.arange(n) - n // 2) * dx_out
    xo, yo = np.meshgrid(x_out, x_out, indexing="xy")

    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field.data)))
    prefactor = (
        np.exp(1j * k * z)
        / (1j * lam * z)
        * np.exp(1j * k * (xo**2 + yo**2) / (2.0 * z))
    )
    # dx**2 makes the DFT approximate the continuous Fourier integral.
    out = prefactor * spectrum * dx**2
    return Field(out, dx_out, lam, z=field.z + z, units=field.units)


def diffraction_reach_px(n: int, dx: float, wavelength: float, z: float) -> float:
    """Farthest, in pixels, one propagation of ``z`` can move energy along one axis.

    A plane-wave component at spatial frequency ``f`` travels at ``sin(theta) =
    lam * f``, so over a distance ``z`` it lands ``z * tan(theta)`` off-axis. The
    grid cannot carry any frequency above Nyquist, ``f_max = 1/(2*dx)``, so no
    pixel can influence one farther away than

        reach = z * lam / (2 * dx^2)    [pixels, paraxial]

    which is this function's return value. It is the *connectivity radius* of a
    single free-space hop: the diffractive counterpart of how many waveguide modes
    one column of a coupler mesh can mix (exactly one neighbour). Stacking ``m``
    hops gives ``m * reach`` -- see ``docs/phase3_mesh.md``.

    Beyond ``z_crit = n * dx^2 / lam`` the Matsushima-Shimobaba band limit in
    :func:`angular_spectrum_transfer` cuts the spectrum below Nyquist, and the
    reach saturates at that lower limit rather than growing with ``z``; this
    function applies the same cut, so it never over-states the reach.

    The exact obliquity form, ``z * lam * f_max / sqrt(1 - (lam*f_max)^2) / dx``,
    is larger by ``(lam/(2*dx))^2 / 2`` in relative terms -- 0.06% at the Phase-2
    operating point. The paraxial value is returned because it is the quotable
    closed form and it errs on the conservative side for a connectivity claim.
    """
    du = 1.0 / (n * dx)
    # Matsushima & Shimobaba (2009), as applied in angular_spectrum_transfer.
    u_limit = 1.0 / (wavelength * np.sqrt((2.0 * du * abs(z)) ** 2 + 1.0))
    f_max = min(1.0 / (2.0 * dx), u_limit)
    return abs(z) * wavelength * f_max / dx


def check_sampling(field: Field, z: float, method: str = "angular_spectrum") -> SamplingReport:
    """Check that ``field`` is adequately sampled to propagate distance ``z``.

    Criteria (Voelz, *Computational Fourier Optics*, ch. 5):

    - ``angular_spectrum`` / ``fresnel`` (transfer-function forms): well sampled
      when ``|z| <= z_crit = n * dx^2 / lam``. Beyond that the transfer function
      under-samples; band-limited ASM suppresses the aliasing but drops
      high-angle content.
    - ``fraunhofer``: valid when ``|z| >= 2 D^2 / lam`` with ``D`` the source
      extent (worst case: the full grid ``n * dx``).

    Returns a :class:`SamplingReport`. Used live in the diffraction explorer and
    wrapped by :func:`photonn.validate.assert_sampling` for runtime enforcement.
    """
    n, dx, lam = field.n, field.dx, field.wavelength
    method = method.lower()
    az = abs(z)

    if method in ("angular_spectrum", "fresnel"):
        z_crit = n * dx**2 / lam
        ok = az <= z_crit
        if ok:
            msg = f"Transfer function well sampled: |z|={az:.4g} m <= z_crit={z_crit:.4g} m."
        else:
            msg = (
                f"|z|={az:.4g} m exceeds z_crit={z_crit:.4g} m: transfer function "
                f"under-sampled. Band-limited ASM suppresses aliasing but discards "
                f"content beyond the propagating band."
            )
        return SamplingReport(ok=ok, method=method, messages=[msg])

    if method == "fraunhofer":
        d = n * dx
        z_fraunhofer = 2.0 * d**2 / lam
        ok = az >= z_fraunhofer
        rel = f"z_Fraunhofer={z_fraunhofer:.4g} m (source extent D={d:.4g} m)"
        if ok:
            msg = f"Far-field valid: |z|={az:.4g} m >= {rel}."
        else:
            msg = f"|z|={az:.4g} m < {rel}: Fraunhofer approximation not yet valid."
        return SamplingReport(ok=ok, method=method, messages=[msg])

    raise ValueError(f"Unknown method {method!r}; use 'angular_spectrum', 'fresnel', or 'fraunhofer'.")
