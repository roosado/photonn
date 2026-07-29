"""One-directional design -> as-built handoff (Python writes, MATLAB reads).

Serializes a trained ideal model, its geometry, its operating point, and the
frozen test set into a single HDF5 file. MATLAB (``photonn-hw/+io/read_handoff.m``)
reads this file and **never writes back** -- the boundary between the ideal
design model and the as-built error model is one-directional by design
(CLAUDE.md handoff contract).

The on-disk layout is specified in ``docs/handoff_schema.md``; this module is
the authoritative writer and validator. Unlike the physics modules, it is fully
implemented -- it is the highest-risk interface and is exercised by a round-trip
test before any physics exists.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import h5py

#: Handoff schema version. Bump on any breaking change to the layout below.
#: The MATLAB reader checks against its own copy of this string.
SCHEMA_VERSION = "0.1.0"

#: Supported model kinds. ``d2nn`` stores phase masks; ``mesh`` stores MZI angles.
MODEL_TYPES = ("d2nn", "mesh")


def _as_str(value):
    """Decode an HDF5 attribute that may come back as bytes into ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def write_handoff(
    path,
    *,
    model_type,
    parameters,
    geometry,
    operating_point,
    test_images,
    test_labels,
    description="",
):
    """Write a handoff HDF5 file. See ``docs/handoff_schema.md`` for the contract.

    Parameters
    ----------
    path : str or os.PathLike
        Output ``.h5`` path (overwritten if it exists).
    model_type : {"d2nn", "mesh"}
        Selects which parameter datasets are written.
    parameters : dict
        ``d2nn`` -> ``{"phase_masks": float[n_layers, N, N]}``.
        ``mesh`` -> ``{"phase_theta": float[...], "phase_phi": float[...]}``.
    geometry : dict
        ``{"grid_size": int, "physical_extent_m": float, "n_layers": int,
        "layer_separations_m": 1D float array}``.
    operating_point : dict
        Scalar operating constants; must include ``"wavelength_m"``. Additional
        keys are written as float attributes on ``/operating_point``.
    test_images : array_like
        Frozen test images, written as ``float32[n, N, N]``.
    test_labels : array_like
        Integer labels, written as ``int32[n]``.
    description : str, optional
        Free-text note stored at the file root.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"model_type must be one of {MODEL_TYPES}; got {model_type!r}."
        )
    if "wavelength_m" not in operating_point:
        raise ValueError("operating_point must include 'wavelength_m'.")
    for key in ("grid_size", "physical_extent_m", "n_layers", "layer_separations_m"):
        if key not in geometry:
            raise ValueError(f"geometry is missing required key {key!r}.")

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["created"] = datetime.now(timezone.utc).isoformat()
        f.attrs["description"] = description

        geo = f.create_group("geometry")
        geo.attrs["grid_size"] = int(geometry["grid_size"])
        geo.attrs["physical_extent_m"] = float(geometry["physical_extent_m"])
        geo.attrs["n_layers"] = int(geometry["n_layers"])
        geo.create_dataset(
            "layer_separations_m",
            data=np.asarray(geometry["layer_separations_m"], dtype="f8"),
        )

        op = f.create_group("operating_point")
        for key, val in operating_point.items():
            op.attrs[key] = float(val)

        p = f.create_group("parameters")
        p.attrs["model_type"] = model_type
        if model_type == "d2nn":
            p.create_dataset(
                "phase_masks", data=np.asarray(parameters["phase_masks"], dtype="f8")
            )
        else:  # mesh
            p.create_dataset(
                "phase_theta", data=np.asarray(parameters["phase_theta"], dtype="f8")
            )
            p.create_dataset(
                "phase_phi", data=np.asarray(parameters["phase_phi"], dtype="f8")
            )

        ts = f.create_group("test_set")
        ts.create_dataset("images", data=np.asarray(test_images, dtype="f4"))
        ts.create_dataset("labels", data=np.asarray(test_labels, dtype="i4"))


def validate_handoff(path):
    """Validate that ``path`` conforms to the handoff schema.

    Reads the file back and asserts that the schema version matches and that all
    required groups, attributes, and datasets are present for the declared
    ``model_type``. Raises :class:`ValueError` on the first violation; returns
    ``None`` on success.
    """
    with h5py.File(path, "r") as f:
        if "schema_version" not in f.attrs:
            raise ValueError("Missing root attribute 'schema_version'.")
        version = _as_str(f.attrs["schema_version"])
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Schema version mismatch: file {version!r}, expected {SCHEMA_VERSION!r}."
            )

        for group in ("geometry", "operating_point", "parameters", "test_set"):
            if group not in f:
                raise ValueError(f"Missing group '/{group}'.")

        geo = f["geometry"]
        for attr in ("grid_size", "physical_extent_m", "n_layers"):
            if attr not in geo.attrs:
                raise ValueError(f"Missing attribute '/geometry.{attr}'.")
        if "layer_separations_m" not in geo:
            raise ValueError("Missing dataset '/geometry/layer_separations_m'.")

        if "wavelength_m" not in f["operating_point"].attrs:
            raise ValueError("Missing attribute '/operating_point.wavelength_m'.")

        params = f["parameters"]
        if "model_type" not in params.attrs:
            raise ValueError("Missing attribute '/parameters.model_type'.")
        model_type = _as_str(params.attrs["model_type"])
        if model_type not in MODEL_TYPES:
            raise ValueError(
                f"'/parameters.model_type' must be one of {MODEL_TYPES}; got {model_type!r}."
            )
        required = ("phase_masks",) if model_type == "d2nn" else ("phase_theta", "phase_phi")
        for dset in required:
            if dset not in params:
                raise ValueError(
                    f"Missing dataset '/parameters/{dset}' for model_type={model_type!r}."
                )

        test_set = f["test_set"]
        for dset in ("images", "labels"):
            if dset not in test_set:
                raise ValueError(f"Missing dataset '/test_set/{dset}'.")
