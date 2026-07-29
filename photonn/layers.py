"""Differentiable (PyTorch) wrappers around the NumPy physics.

Thin by design. Each layer re-expresses a forward pass from
:mod:`photonn.propagate` / :mod:`photonn.mzi` in native ``torch`` operations so
autograd flows through it (FFT-based propagation maps directly onto
``torch.fft``). The NumPy modules remain the validated reference; these layers
are checked against them in the tests. **Autodiff machinery lives here and
nowhere else** -- the physics modules stay pure NumPy (CLAUDE.md convention).

This is the only module in the package that imports ``torch`` at import time.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from photonn.propagate import angular_spectrum_transfer


class PhaseMaskLayer(nn.Module):
    """Trainable phase mask: multiplies the field by ``exp(i * phi)``.

    ``phi`` is a learnable real ``(n, n)`` parameter (radians). Being pure phase,
    the mask is lossless -- ``|E|`` is unchanged -- which is the whole reason a
    D2NN is linear and unitary up to the detector (see ``docs/phase2_dnn.md``).
    Mirrors :func:`photonn.elements.phase_mask`.

    Parameters
    ----------
    n : int
        Grid size; the mask is ``(n, n)``.
    init_std : float, optional
        Std of the zero-mean Gaussian used to initialise ``phi`` (radians).
        Default ``0.0`` gives a transparent (identity) mask, so an untrained
        stack reduces to pure propagation -- deterministic given the run seed.
    """

    def __init__(self, n: int, init_std: float = 0.0):
        super().__init__()
        self.n = n
        if init_std > 0.0:
            phi0 = torch.randn(n, n) * init_std
        else:
            phi0 = torch.zeros(n, n)
        self.phi = nn.Parameter(phi0)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        # exp(i*phi) via polar() keeps the transmittance dtype tied to phi's
        # float dtype (float32 -> complex64) without a python-complex promotion.
        transmittance = torch.polar(torch.ones_like(self.phi), self.phi)
        return field * transmittance.to(field.dtype)


class AngularSpectrumLayer(nn.Module):
    """Fixed-distance angular-spectrum propagation as a differentiable layer.

    The transfer function is precomputed from ``(n, dx, wavelength, z)`` by the
    shared NumPy physics (:func:`photonn.propagate.angular_spectrum_transfer`),
    stored as a non-trainable complex buffer, and applied in the Fourier domain
    via ``torch.fft`` so autograd flows through the field. The layer holds no
    physics of its own -- it only moves the field to the Fourier domain, applies
    the precomputed multiplier, and comes back -- which is what keeps it a thin
    wrapper. Numerically mirrors :func:`photonn.propagate.angular_spectrum`
    (verified in the tests).

    Parameters
    ----------
    n, dx, wavelength, z : grid geometry and propagation distance (metres).
    dtype : torch complex dtype, optional
        ``torch.complex64`` (default) for training; use ``torch.complex128`` to
        match the float64 NumPy reference bit-for-bit in tests.
    """

    def __init__(self, n: int, dx: float, wavelength: float, z: float,
                 dtype: torch.dtype = torch.complex64):
        super().__init__()
        self.n = n
        self.dx = float(dx)
        self.wavelength = float(wavelength)
        self.z = float(z)
        H = angular_spectrum_transfer(n, dx, wavelength, z)  # NumPy complex128, fftfreq order
        self.register_buffer("H", torch.from_numpy(np.ascontiguousarray(H)).to(dtype))

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)))
        out = torch.fft.ifft2(spectrum * self.H)
        return torch.fft.fftshift(out, dim=(-2, -1))


class MZIMeshLayer(nn.Module):
    """A Clements rectangular mesh of MZIs as a differentiable unitary layer.

    Learnable ``(theta, phi)`` per MZI plus an output phase screen. The mesh is a
    brick of ``n_modes`` layers of 2x2 MZIs on alternating adjacent mode pairs
    (``n_modes*(n_modes-1)/2`` MZIs total), the universal rectangular topology of
    Clements et al. (2016). Each MZI uses the same 2x2 block as
    :func:`photonn.mzi.mzi_matrix`; the composed matrix is unitary by construction.

    Forward acts on a ``(batch, n_modes)`` complex tensor. The mesh matrix is
    input-independent, so it is built once per forward and shared across the batch.
    """

    def __init__(self, n_modes: int, init_std: float = None):
        super().__init__()
        self.n = n_modes

        # Clements schedule: layer l couples pairs starting at (l % 2).
        schedule, idx = [], 0
        for layer in range(n_modes):
            offset = layer % 2
            pairs = []
            for m in range(offset, n_modes - 1, 2):
                pairs.append((m, idx))
                idx += 1
            schedule.append(pairs)
        self._schedule = schedule
        self.n_mzi = idx

        # Per-layer index tensors for a vectorised matrix build (top mode of each
        # pair, its MZI index, and the unpaired modes that pass straight through).
        self._layer_index = []
        for pairs in schedule:
            mm = torch.tensor([m for m, _ in pairs], dtype=torch.long)
            ii = torch.tensor([j for _, j in pairs], dtype=torch.long)
            paired = {k for m, _ in pairs for k in (m, m + 1)}
            unp = torch.tensor([k for k in range(n_modes) if k not in paired], dtype=torch.long)
            self._layer_index.append((mm, ii, unp))

        # Random init spreads the mesh over the unitary group (breaks symmetry).
        self.theta = nn.Parameter(torch.rand(idx) * torch.pi)
        self.phi = nn.Parameter(torch.rand(idx) * 2 * torch.pi)
        self.out_phase = nn.Parameter(torch.zeros(n_modes))

    def _blocks(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Vectorised 2x2 MZI blocks ``i e^{i th/2}[[e^{i ph}s, c],[e^{i ph}c, -s]]``.

        ``theta, phi`` are ``(k,)`` phase vectors; returns ``(k, 2, 2)`` complex.
        """
        s = torch.sin(theta / 2)
        c = torch.cos(theta / 2)
        eip = torch.polar(torch.ones_like(phi), phi)              # e^{i phi}
        pre = 1j * torch.polar(torch.ones_like(theta), theta / 2)  # i e^{i theta/2}
        row0 = torch.stack([pre * eip * s, pre * c], dim=-1)
        row1 = torch.stack([pre * eip * c, -pre * s], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    def matrix(self) -> torch.Tensor:
        """Build the ``(n, n)`` complex mesh matrix from the current phases."""
        n, dev = self.n, self.theta.device
        m = torch.eye(n, dtype=torch.complex64, device=dev)
        for mm, ii, unp in self._layer_index:
            mm, ii, unp = mm.to(dev), ii.to(dev), unp.to(dev)
            layer = torch.zeros(n, n, dtype=torch.complex64, device=dev)
            if unp.numel():
                layer[unp, unp] = 1.0
            if mm.numel():
                blk = self._blocks(self.theta[ii], self.phi[ii])   # (k, 2, 2)
                layer[mm, mm] = blk[:, 0, 0]
                layer[mm, mm + 1] = blk[:, 0, 1]
                layer[mm + 1, mm] = blk[:, 1, 0]
                layer[mm + 1, mm + 1] = blk[:, 1, 1]
            m = layer @ m
        out = torch.polar(torch.ones_like(self.out_phase), self.out_phase)
        return torch.diag(out) @ m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix().T   # rows of x are mode vectors
