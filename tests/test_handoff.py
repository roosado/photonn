"""Round-trip tests for the HDF5 handoff contract (photonn/export.py).

These pass today: the handoff serializer is the one fully implemented piece of
the scaffold, so the design->as-built interface is verified before any physics
exists.
"""
from __future__ import annotations

import numpy as np
import h5py
import pytest

from photonn.export import write_handoff, validate_handoff, SCHEMA_VERSION
from photonn.export import _as_str


def test_d2nn_roundtrip(tmp_path, d2nn_payload):
    path = tmp_path / "d2nn.h5"
    write_handoff(path, **d2nn_payload)
    validate_handoff(path)  # must not raise

    with h5py.File(path, "r") as f:
        assert _as_str(f.attrs["schema_version"]) == SCHEMA_VERSION
        assert _as_str(f["parameters"].attrs["model_type"]) == "d2nn"
        np.testing.assert_allclose(
            f["parameters/phase_masks"][...], d2nn_payload["parameters"]["phase_masks"]
        )
        np.testing.assert_array_equal(
            f["test_set/labels"][...], d2nn_payload["test_labels"]
        )
        assert f["geometry"].attrs["grid_size"] == d2nn_payload["geometry"]["grid_size"]
        assert f["operating_point"].attrs["wavelength_m"] == 1.55e-6


def test_mesh_roundtrip(tmp_path, mesh_payload):
    path = tmp_path / "mesh.h5"
    write_handoff(path, **mesh_payload)
    validate_handoff(path)

    with h5py.File(path, "r") as f:
        assert _as_str(f["parameters"].attrs["model_type"]) == "mesh"
        assert "phase_theta" in f["parameters"]
        assert "phase_phi" in f["parameters"]


def test_invalid_model_type_rejected(tmp_path, d2nn_payload):
    payload = dict(d2nn_payload, model_type="nonsense")
    with pytest.raises(ValueError, match="model_type"):
        write_handoff(tmp_path / "bad.h5", **payload)


def test_missing_wavelength_rejected(tmp_path, d2nn_payload):
    payload = dict(d2nn_payload, operating_point={})
    with pytest.raises(ValueError, match="wavelength_m"):
        write_handoff(tmp_path / "bad.h5", **payload)


def test_validate_detects_missing_groups(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
    with pytest.raises(ValueError):
        validate_handoff(path)


def test_validate_detects_schema_mismatch(tmp_path, d2nn_payload):
    path = tmp_path / "d2nn.h5"
    write_handoff(path, **d2nn_payload)
    with h5py.File(path, "a") as f:
        f.attrs["schema_version"] = "9.9.9"
    with pytest.raises(ValueError, match="[Ss]chema version"):
        validate_handoff(path)
