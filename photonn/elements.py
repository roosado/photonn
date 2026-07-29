"""Thin optical elements applied in a field plane.

Each element consumes a :class:`~photonn.fields.Field` and returns a new one,
applying a multiplicative complex transmittance in the plane (thin-element
approximation). Pure and NumPy-native; no autodiff here.
"""
from __future__ import annotations

import numpy as np

from photonn.fields import Field


def _centered_grid(field: Field):
    """Centred-origin coordinate meshes (metres) matching ``field``."""
    x = field.coords()
    return np.meshgrid(x, x, indexing="xy")


def phase_mask(field: Field, phase) -> Field:
    """Apply a spatially varying phase: ``E -> E * exp(i * phase)``.

    ``phase`` is a real ``(N, N)`` array in radians matching the field grid.
    Pure phase, so ``|E|`` is unchanged. This is the NumPy reference for the
    trainable :class:`photonn.layers.PhaseMaskLayer`.
    """
    phase = np.asarray(phase, dtype=float)
    if phase.shape != field.data.shape:
        raise ValueError(
            f"phase must match the field grid {field.data.shape!r}; got {phase.shape!r}."
        )
    transmittance = np.exp(1j * phase)
    return Field(field.data * transmittance, field.dx, field.wavelength, z=field.z, units=field.units)


def amplitude_mask(field: Field, transmission) -> Field:
    """Apply a real amplitude transmittance in ``[0, 1]``: ``E -> E * transmission``.

    ``transmission`` is a real ``(N, N)`` array. Values outside ``[0, 1]`` are
    non-physical for a passive mask and are rejected.
    """
    transmission = np.asarray(transmission, dtype=float)
    if transmission.shape != field.data.shape:
        raise ValueError(
            f"transmission must match the field grid {field.data.shape!r}; got {transmission.shape!r}."
        )
    if transmission.min() < 0.0 or transmission.max() > 1.0:
        raise ValueError(
            f"amplitude transmission must lie in [0, 1]; got range "
            f"[{transmission.min():.3g}, {transmission.max():.3g}]."
        )
    return Field(field.data * transmission, field.dx, field.wavelength, z=field.z, units=field.units)


def thin_lens(field: Field, focal_length: float) -> Field:
    """Apply an ideal thin-lens quadratic phase for the given focal length (m).

    Transmittance ``exp(-i k r^2 / (2 f))``; pure phase, so ``|E|`` is unchanged.
    """
    x, y = _centered_grid(field)
    r2 = x**2 + y**2
    transmittance = np.exp(-1j * field.k * r2 / (2.0 * focal_length))
    return Field(field.data * transmittance, field.dx, field.wavelength, z=field.z, units=field.units)


def aperture(field: Field, shape: str = "circular", size: float = None) -> Field:
    """Apply a hard aperture of the given ``size`` (m).

    ``size`` is the full width: for ``circular`` it is the diameter (radius
    ``size/2``); for ``square`` it is the side length. Transmittance is 1 inside
    and 0 outside.
    """
    if size is None:
        raise ValueError("aperture requires a size (m).")

    x, y = _centered_grid(field)
    shape = shape.lower()
    if shape == "circular":
        radius = size / 2.0
        mask = (x**2 + y**2) <= radius**2
    elif shape == "square":
        half = size / 2.0
        mask = (np.abs(x) <= half) & (np.abs(y) <= half)
    else:
        raise ValueError(f"Unknown aperture shape {shape!r}; use 'circular' or 'square'.")

    return Field(field.data * mask, field.dx, field.wavelength, z=field.z, units=field.units)
