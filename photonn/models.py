"""Network definitions: diffractive (D2NN) and MZI-mesh models.

Both models are deliberately minimal on the electronic side -- the readout is
"integrate intensity, softmax" and nothing more. A larger electronic head is
treated as a finding to document, not a feature to build (CLAUDE.md scope).

Models compose the differentiable layers in :mod:`photonn.layers`; they carry no
physics of their own.
"""
from __future__ import annotations

import torch
from torch import nn

from photonn.detect import default_regions
from photonn.layers import AngularSpectrumLayer, MZIMeshLayer, PhaseMaskLayer


class D2NN(nn.Module):
    """Diffractive deep neural network: a stack of trainable phase masks.

    Forward pass (all propagation steps are equal-distance ``separation``)::

        input -> [propagate -> phase mask] x n_layers -> propagate -> detector

    so there are ``n_layers`` trainable masks and ``n_layers + 1`` fixed
    propagations. At the detector plane the intensity ``|E|^2`` is summed over
    each :class:`~photonn.detect.DetectorRegion` and normalised by the total
    output power, giving a scale-invariant logit per class. The electronic side
    is deliberately just "integrate intensity, softmax" -- the softmax lives in
    the loss (:class:`torch.nn.CrossEntropyLoss`).

    The whole optical path is linear in the input field; the only nonlinearity
    anywhere in the model is the ``|E|^2`` detection. That single square is the
    entire source of the network's expressivity and its ceiling -- see
    ``docs/phase2_dnn.md``.

    Parameters
    ----------
    n, dx, wavelength, separation : grid geometry and inter-mask distance (m).
    n_layers : number of trainable phase masks.
    n_classes : number of detector regions / classes (default 10).
    regions : explicit detector regions; defaults to :func:`default_regions`.
    mask_init_std : std (radians) for phase-mask initialisation (default 0).
    readout_gain : multiplies the normalised region powers to set the softmax
        temperature (default 10.0).
    dtype : propagation buffer dtype (default ``torch.complex64``).
    """

    def __init__(self, n: int, n_layers: int, dx: float, wavelength: float,
                 separation: float, *, n_classes: int = 10, regions=None,
                 mask_init_std: float = 0.0, readout_gain: float = 10.0,
                 dtype: torch.dtype = torch.complex64):
        super().__init__()
        self.n = n
        self.n_layers = n_layers
        self.dx = float(dx)
        self.wavelength = float(wavelength)
        self.separation = float(separation)
        self.n_classes = n_classes
        self.readout_gain = float(readout_gain)

        self.masks = nn.ModuleList(
            PhaseMaskLayer(n, init_std=mask_init_std) for _ in range(n_layers)
        )
        self.props = nn.ModuleList(
            AngularSpectrumLayer(n, dx, wavelength, separation, dtype=dtype)
            for _ in range(n_layers + 1)
        )

        self.regions = regions if regions is not None else default_regions(n, n_classes)
        # One-hot region masks stacked as (n_classes, n, n) for a vectorised readout.
        readout = torch.zeros(len(self.regions), n, n)
        for k, reg in enumerate(self.regions):
            readout[k, reg.y0:reg.y1, reg.x0:reg.x1] = 1.0
        self.register_buffer("readout_masks", readout)

    def output_field(self, field: torch.Tensor) -> torch.Tensor:
        """Propagate ``field`` through the mask stack to the detector plane.

        Returns the complex output field (before detection) -- used for the
        photon budget and mask visualisations in ``docs/phase2_dnn.md``.
        """
        x = field
        for i in range(self.n_layers):
            x = self.props[i](x)
            x = self.masks[i](x)
        return self.props[self.n_layers](x)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        x = self.output_field(field)
        intensity = x.real ** 2 + x.imag ** 2          # |E|^2, (B, n, n); avoids abs() grad kink
        region = torch.einsum("bij,cij->bc", intensity, self.readout_masks)  # (B, C)
        total = intensity.sum(dim=(-2, -1)).clamp_min(1e-12).unsqueeze(-1)   # (B, 1)
        return region / total * self.readout_gain


class MeshNetwork(nn.Module):
    """MZI-mesh classifier, optionally with an SVD (U.Sigma.V*) layer for real matrices.

    With ``use_svd`` (default), the optical transform is ``U · Sigma · V†`` -- two
    universal :class:`~photonn.layers.MZIMeshLayer` unitaries around a diagonal of
    controllable transmissions -- so the mesh can realise an **arbitrary** linear
    map (unlike the D²NN, whose linear map is constrained to phase-masks +
    diffraction). Without it, a single unitary mesh is used. Readout is intensity
    ``|·|²`` on the first ``n_classes`` output modes, normalised by total output
    power (*integrate intensity, softmax* -- no electronic head, same as the D²NN).

    The only nonlinearity is the ``|·|²`` detection; the expressivity discussion in
    ``docs/phase2_dnn.md`` applies here too (see ``docs/phase3_mesh.md``).

    Physical note: a lossless mesh cannot amplify, so a real device needs
    ``Sigma ≤ 1``; in-silico we leave it free and document the constraint.
    """

    def __init__(self, n_modes: int, n_classes: int = 10, *, use_svd: bool = True,
                 readout_gain: float = 10.0):
        super().__init__()
        if n_modes < n_classes:
            raise ValueError(f"n_modes ({n_modes}) must be >= n_classes ({n_classes}).")
        self.n = n_modes
        self.n_classes = n_classes
        self.use_svd = use_svd
        self.readout_gain = float(readout_gain)

        self.u = MZIMeshLayer(n_modes)
        if use_svd:
            self.v = MZIMeshLayer(n_modes)
            self.sigma = nn.Parameter(torch.ones(n_modes))   # diagonal Sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_svd:
            x = self.v(x)
            x = x * self.sigma.to(x.dtype)
        u = self.u(x)
        intensity = u.real ** 2 + u.imag ** 2                # (batch, n_modes)
        region = intensity[:, :self.n_classes]
        total = intensity.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return region / total * self.readout_gain
