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
SCHEMA_VERSION = "0.2.0"

#: Versions a reader accepts. 0.2.0 added the mesh parameters that 0.1.0 left out
#: (Sigma and the output phases); the ``d2nn`` layout did not move, so files written
#: at 0.1.0 -- including the 131 MB ``exports/d2nn_phase2.h5`` -- stay readable
#: without a re-export. See the version history in ``docs/handoff_schema.md``.
SUPPORTED_SCHEMAS = ("0.1.0", "0.2.0")

#: Supported model kinds. ``d2nn`` stores phase masks; ``mesh`` stores MZI angles.
MODEL_TYPES = ("d2nn", "mesh")

#: Mesh topology written into ``/parameters.topology``. The rectangular Clements
#: schedule is the only one the mesh models here use (Optica 3(12):1460, 2016).
MESH_TOPOLOGY = "clements_rectangular"

#: Order the two SVD meshes are concatenated in along ``phase_theta``/``phase_phi``
#: and indexed in along ``out_phase``. The realised operator is ``U diag(s) V``.
MESH_ORDER = "V,U"


def _as_str(value):
    """Decode an HDF5 attribute that may come back as bytes into ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


#: Mesh parameter datasets, in the order they are written and checked.
_MESH_DATASETS = ("phase_theta", "phase_phi", "sigma", "out_phase")


def _mesh_arrays(parameters):
    """Coerce and cross-check the four mesh parameter arrays.

    Returns ``(phase_theta, phase_phi, sigma, out_phase)`` as ``f8``. Raises
    :class:`ValueError` if the shapes cannot describe one consistent SVD mesh --
    the check schema 0.1.0 never made, which is how the exported handoff came to
    be missing 108 of the model's 2 628 parameters without anything noticing.
    """
    for key in _MESH_DATASETS:
        if key not in parameters:
            raise ValueError(f"mesh parameters are missing required key {key!r}.")
    theta = np.asarray(parameters["phase_theta"], dtype="f8")
    phi = np.asarray(parameters["phase_phi"], dtype="f8")
    sigma = np.asarray(parameters["sigma"], dtype="f8")
    out_phase = np.asarray(parameters["out_phase"], dtype="f8")

    if out_phase.ndim != 2:
        raise ValueError(f"'out_phase' must be 2-D [n_meshes, n_modes]; got {out_phase.shape}.")
    n_meshes, n_modes = out_phase.shape
    if sigma.shape != (n_modes,):
        raise ValueError(
            f"'sigma' must be [n_modes]={(n_modes,)}; got {sigma.shape}."
        )
    if theta.shape != phi.shape:
        raise ValueError(
            f"'phase_theta' {theta.shape} and 'phase_phi' {phi.shape} must have the same shape."
        )
    expected = n_meshes * (n_modes * (n_modes - 1) // 2)
    if theta.shape != (expected,):
        raise ValueError(
            f"'phase_theta'/'phase_phi' must be [n_meshes * n_modes(n_modes-1)/2]"
            f"={(expected,)} for {n_meshes} meshes of {n_modes} modes; got {theta.shape}."
        )
    return theta, phi, sigma, out_phase


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
        ``mesh`` -> ``{"phase_theta": float[2 * n_mzi], "phase_phi": float[2 * n_mzi],
        "sigma": float[n_modes], "out_phase": float[2, n_modes]}``, where the two
        meshes are concatenated in :data:`MESH_ORDER` and ``out_phase`` is indexed
        the same way. Together these are everything needed to rebuild the operator;
        schema 0.1.0 carried only the first two and was not sufficient.
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
            theta, phi, sigma, out_phase = _mesh_arrays(parameters)
            n_meshes, n_modes = out_phase.shape
            n_mzi = theta.size // n_meshes
            p.attrs["n_modes"] = int(n_modes)
            p.attrs["n_mzi_per_mesh"] = int(n_mzi)
            p.attrs["mesh_order"] = MESH_ORDER
            p.attrs["topology"] = MESH_TOPOLOGY
            p.create_dataset("phase_theta", data=theta)
            p.create_dataset("phase_phi", data=phi)
            p.create_dataset("sigma", data=sigma)
            p.create_dataset("out_phase", data=out_phase)

        ts = f.create_group("test_set")
        ts.create_dataset("images", data=np.asarray(test_images, dtype="f4"))
        ts.create_dataset("labels", data=np.asarray(test_labels, dtype="i4"))


def validate_handoff(path):
    """Validate that ``path`` conforms to the handoff schema.

    Reads the file back and asserts that the schema version is one this reader
    supports and that all required groups, attributes, and datasets are present
    for the declared ``model_type``. Raises :class:`ValueError` on the first
    violation; returns ``None`` on success.

    A ``mesh`` file at 0.2.0 is additionally checked for shape consistency, so a
    handoff that cannot rebuild its own operator fails here rather than in MATLAB.
    """
    with h5py.File(path, "r") as f:
        if "schema_version" not in f.attrs:
            raise ValueError("Missing root attribute 'schema_version'.")
        version = _as_str(f.attrs["schema_version"])
        if version not in SUPPORTED_SCHEMAS:
            raise ValueError(
                f"Schema version mismatch: file {version!r}, supported {SUPPORTED_SCHEMAS!r}."
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
        if model_type == "d2nn":
            required = ("phase_masks",)
        elif version == "0.1.0":
            # 0.1.0 mesh files carry the MZI angles only. They load, but they cannot
            # rebuild the operator -- see the version history in docs/handoff_schema.md.
            required = ("phase_theta", "phase_phi")
        else:
            required = _MESH_DATASETS
        for dset in required:
            if dset not in params:
                raise ValueError(
                    f"Missing dataset '/parameters/{dset}' for model_type={model_type!r}"
                    f" at schema {version}."
                )
        if model_type == "mesh" and version != "0.1.0":
            _mesh_arrays({k: params[k][...] for k in _MESH_DATASETS})
            for attr in ("n_modes", "n_mzi_per_mesh", "mesh_order", "topology"):
                if attr not in params.attrs:
                    raise ValueError(f"Missing attribute '/parameters.{attr}'.")

        test_set = f["test_set"]
        for dset in ("images", "labels"):
            if dset not in test_set:
                raise ValueError(f"Missing dataset '/test_set/{dset}'.")
