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
from photonn.export import MESH_ORDER, MESH_TOPOLOGY, _as_str


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

    src = mesh_payload["parameters"]
    with h5py.File(path, "r") as f:
        p = f["parameters"]
        assert _as_str(p.attrs["model_type"]) == "mesh"
        # Values, not just presence: schema 0.1.0 shipped a mesh handoff that was
        # missing 108 parameters and nothing caught it, because nothing read them.
        for key in ("phase_theta", "phase_phi", "sigma", "out_phase"):
            np.testing.assert_allclose(p[key][...], src[key])
            assert p[key].dtype == np.dtype("f8")
        n_modes = src["out_phase"].shape[1]
        assert p.attrs["n_modes"] == n_modes
        assert p.attrs["n_mzi_per_mesh"] == n_modes * (n_modes - 1) // 2
        assert _as_str(p.attrs["mesh_order"]) == MESH_ORDER
        assert _as_str(p.attrs["topology"]) == MESH_TOPOLOGY


@pytest.mark.parametrize("missing", ["sigma", "out_phase"])
def test_mesh_export_refuses_to_drop_a_parameter(tmp_path, mesh_payload, missing):
    """The 0.1.0 failure mode -- exporting a mesh that cannot rebuild itself."""
    params = {k: v for k, v in mesh_payload["parameters"].items() if k != missing}
    payload = dict(mesh_payload, parameters=params)
    with pytest.raises(ValueError, match=missing):
        write_handoff(tmp_path / "lossy.h5", **payload)


def test_mesh_export_refuses_inconsistent_shapes(tmp_path, mesh_payload):
    params = dict(mesh_payload["parameters"], sigma=np.zeros(3))
    with pytest.raises(ValueError, match="sigma"):
        write_handoff(tmp_path / "bad.h5", **dict(mesh_payload, parameters=params))


def test_a_schema_0_1_0_file_still_validates(tmp_path, d2nn_payload):
    """0.2.0 was additive and mesh-only, so old d2nn handoffs must keep loading.

    ``exports/d2nn_phase2.h5`` is 131 MB and regenerating it means a retrain; the
    error budget it anchors is already published.
    """
    path = tmp_path / "old.h5"
    write_handoff(path, **d2nn_payload)
    with h5py.File(path, "a") as f:
        f.attrs["schema_version"] = "0.1.0"
    validate_handoff(path)  # must not raise


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
