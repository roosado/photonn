"""Shared pytest fixtures.

Provides ready-made, schema-valid handoff payloads so tests can exercise the
serializer without restating the contract each time.
"""
from __future__ import annotations

import numpy as np
import pytest

from photonn.detect import default_regions


@pytest.fixture
def rng():
    """A seeded NumPy Generator (seeds fixed and recorded per project convention)."""
    return np.random.default_rng(20260723)


def _geometry(grid_size, n_layers, *, regions=True):
    geo = {
        "grid_size": grid_size,
        "physical_extent_m": 1.0e-3,
        "n_layers": n_layers,
        "layer_separations_m": np.full(n_layers, 3.0e-2, dtype="f8"),
    }
    if regions:
        # Schema 0.3.0: the layout crosses the seam as data instead of being
        # re-derived from fractions typed into the MATLAB port.
        geo["detector_regions"] = default_regions(grid_size, 10)
    return geo


def _test_set(rng, grid_size, n_samples=5):
    images = rng.random((n_samples, grid_size, grid_size)).astype("f4")
    labels = (np.arange(n_samples) % 10).astype("i4")
    return images, labels


#: A complete operating point per model kind, matching what the training scripts
#: actually write. These fixtures used to carry ``wavelength_m`` alone, which was
#: all ``validate_handoff`` checked -- so "schema-valid payload" meant valid
#: against a check that covered one field of eleven. Now that the manifest in
#: ``photonn.export.OPERATING_POINT`` is enforced at write time, a fixture has to
#: be as complete as the thing it stands in for, which is the point of a fixture.
_OPERATING_POINT = {
    "d2nn": {
        "wavelength_m": 1.55e-6,
        "pixel_pitch_m": 8.0e-6,
        "readout_gain": 10.0,
        "phase_scale_rad": float(np.pi),
        "input_frac": 0.5,
        "encoding_code": 2,          # "both"
        "input_power_w": 1.0e-3,
        "integration_time_s": 1.0e-3,
    },
    "mesh": {
        "wavelength_m": 1.55e-6,
        "readout_gain": 1.0,
        "n_modes": 4,
        "n_classes": 10,
        "sigma_gain": 1.0,
        "input_power_w": 1.0e-3,
        "integration_time_s": 1.0e-3,
    },
}


@pytest.fixture
def d2nn_payload(rng):
    """A valid ``d2nn`` handoff payload as keyword args for ``write_handoff``."""
    grid_size, n_layers = 8, 3
    images, labels = _test_set(rng, grid_size)
    return dict(
        model_type="d2nn",
        parameters={"phase_masks": rng.random((n_layers, grid_size, grid_size))},
        geometry=_geometry(grid_size, n_layers),
        operating_point=dict(_OPERATING_POINT["d2nn"]),
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
        geometry=_geometry(grid_size, n_meshes, regions=False),
        operating_point=dict(_OPERATING_POINT["mesh"], n_modes=n_modes),
        test_images=images,
        test_labels=labels,
        description="mesh round-trip fixture",
    )


def _deep_operating_point(n_modes):
    """A consistent Phase-5 operating point: g_phi and phi_b follow from the raw values."""
    alpha, gain_ohm, resp, v_pi, v_b = 0.1, 5000.0, 1.0, 10.0, 10.0
    return dict(
        _OPERATING_POINT["mesh"], n_modes=n_modes,
        eo_alpha=alpha, eo_tia_gain_ohm=gain_ohm, eo_responsivity_a_per_w=resp,
        eo_v_pi=v_pi, eo_v_bias=v_b,
        eo_g_phi=float(np.pi * alpha * gain_ohm * resp / v_pi),
        eo_phi_b=float(np.pi * v_b / v_pi),
        eo_bandwidth_hz=1.0e10,
    )


@pytest.fixture
def deep_mesh_payload(rng):
    """A valid ``deep_mesh`` handoff payload (schema 0.4.0): two layers of four modes."""
    n_layers, n_modes = 2, 4
    n_mzi = n_modes * (n_modes - 1) // 2
    images, labels = _test_set(rng, 28)
    inputs = rng.standard_normal((len(labels), n_modes)) + 1j * rng.standard_normal(
        (len(labels), n_modes))
    inputs /= np.linalg.norm(inputs, axis=1, keepdims=True)
    return dict(
        model_type="deep_mesh",
        parameters={
            "phase_theta": rng.random((n_layers, 2 * n_mzi)),
            "phase_phi": rng.random((n_layers, 2 * n_mzi)),
            "sigma": rng.random((n_layers, n_modes)),
            "out_phase": rng.random((n_layers, 2, n_modes)),
        },
        geometry=_geometry(n_modes, 2 * n_layers, regions=False),
        operating_point=_deep_operating_point(n_modes),
        test_images=images,
        test_labels=labels,
        test_inputs=inputs,
        description="deep_mesh round-trip fixture",
    )
