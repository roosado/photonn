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


def _enforce_sampling(field: Field, z: float, method: str) -> None:
    """Enforce the sampling criterion, unless the caller opted out.

    Imported lazily: :mod:`photonn.validate` imports :func:`check_sampling` from
    here, so a module-level import would be circular.

    This is the line that makes "sampling checks as runtime assertions, not just
    tests" true. It was not: every propagator returned a quietly aliased field
    past ``z_crit`` and left the caller to notice. Opt out with
    ``photonn.validate.relaxed()`` where being outside the criterion is the point.
    """
    from photonn import validate

    if validate.STRICT:
        validate.assert_sampling(field, z, method)


def angular_spectrum(field: Field, z: float) -> Field:
    """Propagate ``field`` a distance ``z`` by the angular-spectrum method.

    Exact for scalar, monochromatic fields up to the band limit of the grid --
    no paraxial approximation. Evanescent components decay; the transfer function
    is band-limited per Matsushima & Shimobaba (2009) so the method stays
    alias-free into the far field (at the cost of discarding content beyond the
    propagating band). Reference for :func:`fresnel`, :func:`fraunhofer`, and the
    torch layer in :mod:`photonn.layers`.
    """
    _enforce_sampling(field, z, "angular_spectrum")
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
    _enforce_sampling(field, z, "fresnel")
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
    _enforce_sampling(field, z, "fraunhofer")
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


def required_reach_px(n: int, regions, *, input_frac: float = 0.5) -> float:
    """Total reach the stack needs before it can compute the mapping at all.

    Worst case, per axis: the input pixel at one edge of the entrance window must
    be able to influence the detector pixel farthest from it. Below this the
    failure is geometric rather than statistical -- part of the digit physically
    cannot reach the detector that needs it, whatever the masks say. Compare the
    answer against ``m * diffraction_reach_px(...)`` for an ``m``-hop stack.

    Pure geometry: it needs the grid, the entrance window and where the detectors
    sit, and no trained model. That is deliberate. This derivation previously
    lived in :mod:`apps.export_analogy_web`, which reads a trained handoff, so a
    second copy had to be kept in :mod:`apps.sweep_optics` to answer the same
    question for a grid nothing had been trained on -- and two further copies
    drifted into a plotting script and a test as the bare literal ``74.0``.
    Nothing needed a second implementation; it needed this one not to require a
    handoff.

    Scales with the grid, because ``detect.default_regions`` and
    ``train._embed`` both place things as *fractions* of ``n``: a larger grid is
    a proportionally larger device needing proportionally more reach. That is
    exactly why the wrap budget alone does not answer the grid question.
    """
    return float(max(required_reach_px_axes(n, regions, input_frac=input_frac)))


def required_reach_px_axes(n: int, regions, *, input_frac: float = 0.5):
    """``(need_x, need_y)`` for :func:`required_reach_px`.

    Reach is per-axis -- the FFT band is a square in ``(fx, fy)`` -- so x and y
    are separate one-dimensional tests, and they differ whenever the detector
    lattice is not square (at 128 with ten classes: 74 and 70).
    """
    win = max(1, int(round(input_frac * n)))
    off = (n - win) // 2
    lo, hi = off, off + win - 1                              # inclusive indices
    det_x0 = min(r.x0 for r in regions)
    det_x1 = max(r.x1 - 1 for r in regions)
    det_y0 = min(r.y0 for r in regions)
    det_y1 = max(r.y1 - 1 for r in regions)
    return (float(max(hi - det_x0, det_x1 - lo)),
            float(max(hi - det_y0, det_y1 - lo)))


def stack_wraparound_error(fields, z: float, n_hops: int, regions, *,
                           pad_factor: int = 3) -> dict:
    """Wrap error accumulated over ``n_hops`` hops, at the plane and at the detectors.

    Single-hop :func:`wraparound_error` understates a D2NN badly: the model takes
    ``n_layers + 1`` hops and the wrapped energy compounds. This is the stacked
    form, and it is what gates a geometry sweep -- a configuration over the wrap
    budget is not trained, so the criterion has to run before any model exists.

    What the classifier actually consumes is not the field but ten
    region-integrated intensities, and those are far more forgiving: the patches
    sit in the central 75 % of the grid, while wrapped energy arrives at the
    edges. Both are reported because the gap between them *is* the finding.

    Masks are omitted (identity), so this is pure geometry and needs no trained
    model. Returns ``plane_error`` and ``logit_error`` as mean relative residuals
    against a ``pad_factor``-padded reference, ``argmax_flips`` as the count of
    probe fields whose predicted class the wrap actually moves, and ``n_probe``.
    """
    from photonn.detect import integrate_intensity

    fields = list(fields)
    if not fields:
        raise ValueError("stack_wraparound_error needs at least one probe field.")
    if int(pad_factor) < 2:
        raise ValueError(
            f"pad_factor must be >= 2 to leave room for the spread; got {pad_factor!r}."
        )

    # Same reason as wraparound_error: the stacked wrap criterion is measured
    # past the point where the grid is comfortable, which is precisely the
    # regime a geometry sweep is asking about.
    from photonn.validate import relaxed

    def chain(f: Field) -> Field:
        with relaxed():
            for _ in range(n_hops):
                f = angular_spectrum(f, z)
        return f

    plane, logit, flips = [], [], 0
    for f in fields:
        n, m = f.n, int(pad_factor) * f.n
        lo = (m - n) // 2

        on_grid = chain(f)

        big = np.zeros((m, m), dtype=complex)
        big[lo:lo + n, lo:lo + n] = f.data
        wide = chain(Field(big, f.dx, f.wavelength, z=f.z, units=f.units))
        reference = Field(wide.data[lo:lo + n, lo:lo + n], f.dx, f.wavelength,
                          z=wide.z, units=f.units)

        denom = np.linalg.norm(reference.data)
        plane.append(float(np.linalg.norm(on_grid.data - reference.data) / denom)
                     if denom else 0.0)

        # Through the same readout the model uses, rather than a second inline
        # sum over the same slices: `readout_gain` and the region layout mean
        # something only if every consumer integrates the same way.
        a = integrate_intensity(on_grid, regions)
        b = integrate_intensity(reference, regions)
        nb = np.linalg.norm(b)
        logit.append(float(np.linalg.norm(a - b) / nb) if nb else 0.0)
        flips += int(np.argmax(a) != np.argmax(b))

    return {"plane_error": float(np.mean(plane)), "logit_error": float(np.mean(logit)),
            "argmax_flips": flips, "n_probe": len(fields)}


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


def wraparound_error(field: Field, z: float, *, pad_factor: int = 2) -> float:
    """Relative error that circular FFT wrap-around adds to one propagation.

    :func:`angular_spectrum` propagates on the field's own grid with a plain
    ``fft2``, which is **periodic**: energy that diffracts past one edge reappears
    on the opposite one, as if the aperture were tiled infinitely. This is a
    different failure from the one :func:`check_sampling` reports -- that criterion
    asks whether the transfer function is sampled finely enough (``|z| <= z_crit``)
    and says nothing about whether the *window* is wide enough to contain the
    spread. A field can pass ``check_sampling`` comfortably and still be wrapping.

    Propagating the same field on a ``pad_factor * n`` grid of zeros and cropping
    back to the centre gives the wrap-free reference -- the padding absorbs the
    energy that would otherwise have re-entered. This returns

        ||E_grid - E_padded|| / ||E_padded||

    over the original window: ``0`` means the grid is wide enough at this ``z``,
    and order ``1`` means the simulation describes a periodically tiled system
    rather than free space.

    Within ``|z| <= z_crit`` the Matsushima band limit is inactive on both grids
    (padding lowers ``du``, which only relaxes it further) and both carry the full
    Nyquist band ``1/(2*dx)`` set by the shared pitch, so the residual is
    wrap alone. Beyond ``z_crit`` the two grids band-limit differently and the
    number conflates wrap with that truncation; it is then an upper bound on wrap
    rather than a measurement of it.

    **Diagnostic only.** It deliberately does not change :func:`angular_spectrum`:
    padding the propagator would move every already-published result that was
    computed without it.
    """
    pad_factor = int(pad_factor)
    if pad_factor < 2:
        raise ValueError(f"pad_factor must be >= 2 to leave room for the spread; got {pad_factor!r}.")

    n = field.n
    m = pad_factor * n
    lo = (m - n) // 2

    padded = np.zeros((m, m), dtype=complex)
    padded[lo:lo + n, lo:lo + n] = field.data

    # Deliberately outside the sampling criterion: this function is a wrap
    # diagnostic, and its docstring already states what the number means past
    # z_crit (an upper bound on wrap, conflated with band-limit truncation).
    # Enforcing sampling here would refuse the measurement being asked for.
    from photonn.validate import relaxed

    with relaxed():
        reference = angular_spectrum(
            Field(padded, field.dx, field.wavelength), z).data[lo:lo + n, lo:lo + n]
        on_grid = angular_spectrum(field, z).data

    denom = np.linalg.norm(reference)
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(on_grid - reference) / denom)
