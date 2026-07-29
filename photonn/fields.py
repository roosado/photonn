"""Complex optical field objects with physical units.

A :class:`Field` bundles a complex scalar field, sampled on a square uniform
grid, together with its physical sample spacing, wavelength, and axial position.
Every physics function in the package consumes and returns ``Field`` objects --
no bare arrays cross module boundaries (CLAUDE.md working convention). This unit
bookkeeping is the foundation the rest of the project rests on, so it is
implemented (and tested) here rather than stubbed.

Units: SI throughout. All lengths in metres.
"""
from __future__ import annotations

import numpy as np


class Field:
    """A scalar complex field on a square, uniformly sampled grid.

    Parameters
    ----------
    data : array_like
        Square ``(N, N)`` complex field amplitude. Real input is promoted to
        complex.
    dx : float
        Sample spacing (grid pitch) in metres. Must be positive.
    wavelength : float
        Free-space wavelength in metres. Must be positive.
    z : float, optional
        Axial position of this field plane in metres (default ``0.0``).
    units : str, optional
        Human-readable note on the unit convention (default SI/metres).

    Notes
    -----
    The array is stored row-major with shape ``(N, N)``; index order is
    ``data[iy, ix]``. Physical coordinates run through zero at the grid centre
    (see :meth:`coords`).
    """

    def __init__(self, data, dx, wavelength, z=0.0, units="SI (metres)"):
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise ValueError(
                f"Field data must be a square 2D array; got shape {data.shape!r}."
            )
        if not np.iscomplexobj(data):
            data = data.astype(np.complex128)

        dx = float(dx)
        wavelength = float(wavelength)
        if dx <= 0.0:
            raise ValueError(f"Sample spacing dx must be positive; got {dx!r}.")
        if wavelength <= 0.0:
            raise ValueError(f"Wavelength must be positive; got {wavelength!r}.")

        self.data = data
        self.dx = dx
        self.wavelength = wavelength
        self.z = float(z)
        self.units = units

    # -- grid / geometry -------------------------------------------------
    @property
    def n(self) -> int:
        """Number of samples per side (grid is ``n`` x ``n``)."""
        return self.data.shape[0]

    @property
    def extent(self) -> float:
        """Physical side length of the grid in metres (``n * dx``)."""
        return self.n * self.dx

    @property
    def k(self) -> float:
        """Free-space wavenumber ``2*pi/wavelength`` (rad/m)."""
        return 2.0 * np.pi / self.wavelength

    def coords(self) -> np.ndarray:
        """Centred 1D coordinate axis in metres, length ``n``.

        The same axis applies to both grid dimensions.
        """
        n = self.n
        return (np.arange(n) - n // 2) * self.dx

    # -- field readouts --------------------------------------------------
    def intensity(self) -> np.ndarray:
        """Intensity ``|E|**2`` as a real array (arbitrary units)."""
        return np.abs(self.data) ** 2

    def phase(self) -> np.ndarray:
        """Wrapped phase ``arg(E)`` in radians, range ``(-pi, pi]``."""
        return np.angle(self.data)

    def power(self) -> float:
        """Total power: intensity integrated over the grid (``sum |E|**2 * dx**2``)."""
        return float(np.sum(self.intensity()) * self.dx**2)

    # -- utilities -------------------------------------------------------
    def copy(self) -> "Field":
        """Return a deep copy (the underlying array is duplicated)."""
        return Field(self.data.copy(), self.dx, self.wavelength, self.z, self.units)

    def __repr__(self) -> str:
        return (
            f"Field(n={self.n}, dx={self.dx:.3e} m, extent={self.extent:.3e} m, "
            f"wavelength={self.wavelength:.3e} m, z={self.z:.3e} m)"
        )
