"""Detector regions, photon budget, and detector noise.

Defines integrated-intensity readout over detector regions, establishes the
optical power budget (photons per detector region per inference), and models the
detector noise that feeds the Phase-4 error budget: shot noise (from the photon
budget), thermal noise, and ADC quantization.

Physical constants introduced here must be cited inline or marked ``# UNSOURCED``
(see ``docs/parameter_sources.md``). Pure and NumPy-native.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as _dc_field

import numpy as np

from photonn.fields import Field

# Exact defining constants of the SI (2019 redefinition); not measured values,
# so no measurement source is needed beyond the SI definition itself.
H_PLANCK = 6.62607015e-34   # Planck constant (J*s), exact (SI 2019 definition)
C_LIGHT = 299_792_458.0     # speed of light in vacuum (m/s), exact (SI definition)


class DetectorRegion:
    """A rectangular detector region over which intensity is integrated.

    Bounds are given in grid-index units (half-open, like NumPy slicing:
    rows ``y0:y1``, columns ``x0:x1``). The class-to-region mapping is fixed per
    task (see :func:`default_regions`) and reused across phases so results stay
    comparable.
    """

    def __init__(self, y0: int, y1: int, x0: int, x1: int, label=None):
        y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
        if y1 <= y0 or x1 <= x0:
            raise ValueError(f"empty region: (y0,y1,x0,x1)=({y0},{y1},{x0},{x1}).")
        if y0 < 0 or x0 < 0:
            raise ValueError(f"region bounds must be non-negative: ({y0},{y1},{x0},{x1}).")
        self.y0, self.y1, self.x0, self.x1 = y0, y1, x0, x1
        self.label = label

    @property
    def slices(self):
        """``(row_slice, col_slice)`` for indexing an ``(n, n)`` array."""
        return (slice(self.y0, self.y1), slice(self.x0, self.x1))

    @property
    def area_px(self) -> int:
        """Region area in pixels."""
        return (self.y1 - self.y0) * (self.x1 - self.x0)

    def __repr__(self) -> str:
        return (f"DetectorRegion(y0={self.y0}, y1={self.y1}, x0={self.x0}, "
                f"x1={self.x1}, label={self.label!r})")


def default_regions(n: int, n_classes: int = 10, *, cols: int = None,
                    field_frac: float = 0.75, patch_frac: float = 0.11):
    """Fixed detector layout: ``n_classes`` square patches on a regular lattice.

    The patches tile the central ``field_frac`` of the grid in a ``rows x cols``
    arrangement (one class per cell, in row-major order), each patch of side
    ``patch_frac * n`` pixels centred in its cell. Non-overlapping as long as the
    patch is smaller than a cell. Deterministic, so the same mapping is reused
    across phases and by the MATLAB as-built model.
    """
    if cols is None:
        cols = math.ceil(math.sqrt(n_classes))
    rows = math.ceil(n_classes / cols)
    patch = max(1, int(round(patch_frac * n)))
    span = field_frac * n
    origin = (n - span) / 2.0

    regions = []
    for idx in range(n_classes):
        r, c = divmod(idx, cols)
        cy = origin + (r + 0.5) * span / rows
        cx = origin + (c + 0.5) * span / cols
        y0 = int(round(cy - patch / 2.0))
        x0 = int(round(cx - patch / 2.0))
        y0 = min(max(y0, 0), n - patch)
        x0 = min(max(x0, 0), n - patch)
        regions.append(DetectorRegion(y0, y0 + patch, x0, x0 + patch, label=idx))
    return regions


def integrate_intensity(field: Field, regions) -> np.ndarray:
    """Integrate field intensity over each region; return a readout vector.

    Returns ``sum |E|^2 * dx^2`` per region, i.e. optical power in the same
    (arbitrary) units as :meth:`photonn.fields.Field.power`. This is the NumPy
    reference for the differentiable readout in :class:`photonn.models.D2NN`.
    """
    inten = field.intensity()
    dA = field.dx ** 2
    return np.array([inten[reg.slices].sum() * dA for reg in regions], dtype=float)


@dataclass
class PhotonBudget:
    """Optical power budget for one inference.

    Attributes
    ----------
    photons_in : float
        Total photons delivered per inference, ``P_in * t_int / E_photon``.
    photon_energy_j : float
        Single-photon energy ``h c / lambda`` (J).
    per_pixel : np.ndarray
        ``(n, n)`` map of photons per pixel at this plane.
    per_region : np.ndarray | None
        Photons per detector region (if regions were supplied).
    captured_fraction : float | None
        Fraction of delivered photons landing inside the detector regions.
    """

    photons_in: float
    photon_energy_j: float
    per_pixel: np.ndarray = _dc_field(repr=False)
    per_region: np.ndarray = None
    captured_fraction: float = None


def photon_budget(field: Field, *, input_power_w: float, integration_time_s: float,
                  regions=None, reference_power: float = 1.0) -> PhotonBudget:
    """Photons per detector region per inference, from the optical power budget.

    The delivered energy per inference is ``E_in = input_power_w *
    integration_time_s``; each photon carries ``E_photon = h c / lambda``, so
    ``N_in = E_in / E_photon`` photons enter the system. The field's intensity is
    in arbitrary units, converted to absolute photon counts under the convention
    that ``reference_power`` (default ``1.0``) is the total power of the *input*
    field -- so if the input is power-normalised to 1 (as
    :func:`photonn.train.encode_input` does), a downstream field's own reduced
    power automatically accounts for propagation/aperture losses. Passing that
    same normalised input field back here reproduces the full ``N_in``.
    """
    e_photon = H_PLANCK * C_LIGHT / field.wavelength
    n_in = input_power_w * integration_time_s / e_photon

    scale = n_in / reference_power
    per_pixel = field.intensity() * field.dx ** 2 * scale

    per_region = None
    captured_fraction = None
    if regions is not None:
        per_region = np.array([per_pixel[reg.slices].sum() for reg in regions], dtype=float)
        captured_fraction = float(per_region.sum() / n_in) if n_in > 0 else 0.0

    return PhotonBudget(
        photons_in=float(n_in),
        photon_energy_j=float(e_photon),
        per_pixel=per_pixel,
        per_region=per_region,
        captured_fraction=captured_fraction,
    )


def shot_noise(readout, *, seed: int):
    """Apply Poisson shot noise to a photon-count readout (recorded ``seed``)."""
    raise NotImplementedError("Phase 4: shot noise not yet implemented.")


def thermal_noise(readout, *, noise_power, seed: int):
    """Add detector thermal (Johnson) noise to a readout (recorded ``seed``)."""
    raise NotImplementedError("Phase 4: thermal noise not yet implemented.")
