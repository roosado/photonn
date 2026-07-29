"""photonn — physical photonic neural network simulator (design side).

Pure-NumPy scalar wave-optics physics plus thin PyTorch wrappers. Trains an
idealized optical classifier and serializes it across the one-directional HDF5
handoff to the MATLAB as-built error model (see ``photonn-hw/``).

The package core (:class:`~photonn.fields.Field` and the handoff serializer) is
NumPy/h5py only. The differentiable layers in :mod:`photonn.layers` pull in
PyTorch and are imported on demand, not at package import time.
"""
from __future__ import annotations

from photonn.fields import Field
from photonn.export import SCHEMA_VERSION, write_handoff, validate_handoff

__version__ = "0.1.0"

__all__ = [
    "Field",
    "SCHEMA_VERSION",
    "write_handoff",
    "validate_handoff",
    "__version__",
]
