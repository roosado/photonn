"""Shared pytest fixtures.

Provides ready-made, schema-valid handoff payloads so tests can exercise the
serializer without restating the contract each time.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    """A seeded NumPy Generator (seeds fixed and recorded per project convention)."""
    return np.random.default_rng(20260723)


def _geometry(grid_size, n_layers):
    return {
        "grid_size": grid_size,
        "physical_extent_m": 1.0e-3,
        "n_layers": n_layers,
        "layer_separations_m": np.full(n_layers, 3.0e-2, dtype="f8"),
    }


def _test_set(rng, grid_size, n_samples=5):
    images = rng.random((n_samples, grid_size, grid_size)).astype("f4")
    labels = (np.arange(n_samples) % 10).astype("i4")
    return images, labels


@pytest.fixture
def d2nn_payload(rng):
    """A valid ``d2nn`` handoff payload as keyword args for ``write_handoff``."""
    grid_size, n_layers = 8, 3
    images, labels = _test_set(rng, grid_size)
    return dict(
        model_type="d2nn",
        parameters={"phase_masks": rng.random((n_layers, grid_size, grid_size))},
        geometry=_geometry(grid_size, n_layers),
        operating_point={"wavelength_m": 1.55e-6},
        test_images=images,
        test_labels=labels,
        description="d2nn round-trip fixture",
    )


@pytest.fixture
def mesh_payload(rng):
    """A valid ``mesh`` handoff payload as keyword args for ``write_handoff``.

    Two meshes (V and U) of ``n_modes``, matching the SVD layer the Phase-3 model
    uses, so the shape cross-checks in ``export._mesh_arrays`` are exercised.
    """
    grid_size, n_modes, n_meshes = 8, 4, 2
    n_mzi = n_meshes * (n_modes * (n_modes - 1) // 2)
    images, labels = _test_set(rng, grid_size)
    return dict(
        model_type="mesh",
        parameters={
            "phase_theta": rng.random(n_mzi),
            "phase_phi": rng.random(n_mzi),
            "sigma": rng.random(n_modes),
            "out_phase": rng.random((n_meshes, n_modes)),
        },
        geometry=_geometry(grid_size, n_meshes),
        operating_point={"wavelength_m": 1.55e-6},
        test_images=images,
        test_labels=labels,
        description="mesh round-trip fixture",
    )
