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
from typing import NamedTuple

import numpy as np
import h5py

#: Handoff schema version. Bump on any breaking change to the layout below.
#: The MATLAB reader checks against its own copy of this string.
SCHEMA_VERSION = "0.4.0"

#: Versions a reader accepts. 0.2.0 added the mesh parameters that 0.1.0 left out
#: (Sigma and the output phases); the ``d2nn`` layout did not move, so files written
#: at 0.1.0 -- including the 131 MB ``exports/d2nn_phase2.h5`` -- stay readable
#: without a re-export. 0.3.0 adds ``/geometry/detector_regions``: where the
#: detectors sit was the one design parameter the handoff never carried, so MATLAB
#: re-derived it from re-typed fractions and agreed with Python only because
#: someone kept the two arithmetics in step. Older files carry no regions and
#: readers fall back to deriving them, which is what they were doing anyway.
#: 0.4.0 adds the ``deep_mesh`` kind (Phase 5: SVD layers with an electro-optic
#: activation between them, and its constants on ``/operating_point``) and the
#: encoded test inputs as data (``/test_set/inputs_re``, ``inputs_im``), because the
#: Phase-5 input is complex and a magnitude map cannot carry it. Additive: no
#: ``d2nn`` or ``mesh`` layout moves.
#: See the version history in ``docs/handoff_schema.md``.
SUPPORTED_SCHEMAS = ("0.1.0", "0.2.0", "0.3.0", "0.4.0")

#: Supported model kinds. ``d2nn`` stores phase masks; ``mesh`` stores MZI angles;
#: ``deep_mesh`` stores one ``mesh`` parameter block per layer.
MODEL_TYPES = ("d2nn", "mesh", "deep_mesh")

#: Mesh topology written into ``/parameters.topology``. The rectangular Clements
#: schedule is the only one the mesh models here use (Optica 3(12):1460, 2016).
MESH_TOPOLOGY = "clements_rectangular"

#: Order the two SVD meshes are concatenated in along ``phase_theta``/``phase_phi``
#: and indexed in along ``out_phase``. The realised operator is ``U diag(s) V``.
MESH_ORDER = "V,U"


class OperatingPointField(NamedTuple):
    """One scalar constant the handoff may carry."""

    required_for: tuple
    doc: str


#: Every scalar ``/operating_point`` is allowed to hold, which model kinds require
#: it, and what it means -- the field manifest for the project's most important
#: seam.
#:
#: This exists because the seam used to be enforced at one end and depended on at
#: six. ``write_handoff`` wrote whatever dict it was handed, ``validate_handoff``
#: checked exactly one key (``wavelength_m``), and the schema doc formalised the
#: hole with "additional scalar constants may be added". Meanwhile eleven keys
#: were load-bearing downstream and every reader re-stated the subset it needed by
#: literal string, including a MATLAB reader that turned absence into a *default*.
#:
#: The failure that shape produces is the worst kind available here: rename
#: ``pixel_pitch_m`` and the writer accepted it, the validator passed, the tests
#: passed, and MATLAB propagated ``NaN``. Rename ``readout_gain`` and MATLAB
#: silently substituted 1.0 for 10.0 -- logits rescaled tenfold, no error
#: anywhere, into a published tolerance number.
#:
#: With the manifest, an unknown key fails at write time and a missing one fails
#: before the file exists, so what a reader receives is correct by construction.
#: Adding a constant means adding a row here; that is the whole cost.
OPERATING_POINT = {
    "wavelength_m": OperatingPointField(
        ("d2nn", "mesh", "deep_mesh"), "Operating wavelength."),
    "readout_gain": OperatingPointField(
        ("d2nn", "mesh", "deep_mesh"),
        "Scale from normalised region intensities to logits. Wrong value "
        "rescales every logit and still classifies, so nothing downstream "
        "notices."),
    "input_power_w": OperatingPointField(
        ("d2nn", "mesh", "deep_mesh"),
        "Optical power at the entrance; half the photon budget. For deep_mesh it is "
        "also where the activation sits on its curve, so a wrong value changes what "
        "the network computes, not only how noisy it is."),
    "integration_time_s": OperatingPointField(
        ("d2nn", "mesh", "deep_mesh"),
        "Detector integration window; with input_power_w gives photons per "
        "inference, which is what the shot-noise sweep is denominated in."),
    "pixel_pitch_m": OperatingPointField(
        ("d2nn",),
        "Grid pitch. The transfer function goes as 1/dx^2, so an error here is "
        "squared into every propagation."),
    "phase_scale_rad": OperatingPointField(
        ("d2nn",), "Full-scale phase one mask pixel can apply."),
    "input_frac": OperatingPointField(
        ("d2nn",), "Fraction of the grid the encoded digit is embedded into."),
    "encoding_code": OperatingPointField(
        ("d2nn",),
        "0 amplitude, 1 phase, 2 both. Reconstructing the input under the wrong "
        "scheme produces a plausible field that is not the trained one."),
    "n_modes": OperatingPointField(("mesh", "deep_mesh"), "Mesh width."),
    "n_classes": OperatingPointField(("mesh", "deep_mesh"), "Readout classes."),
    "sigma_gain": OperatingPointField(
        ("mesh", "deep_mesh"),
        "External gain undoing the passivization of sigma; logit-preserving only "
        "if applied. For deep_mesh it belongs to the last layer only: every sigma "
        "ahead of an activation is passive by construction."),
    # -- Phase 5: the electro-optic activation (Williamson et al. 2020) ----------
    # The derived three are what the forward pass uses; the raw device values are
    # what an as-built model perturbs. Both cross, and the writer checks that the
    # raw values reproduce the derived ones (Eqs. 5 and 7).
    "eo_alpha": OperatingPointField(
        ("deep_mesh",), "Power fraction tapped to the activation's photodiode."),
    "eo_g_phi": OperatingPointField(
        ("deep_mesh",),
        "Phase gain, rad/W: pi alpha G R / V_pi (Eq. 7). Multiplies |z|^2 in watts."),
    "eo_phi_b": OperatingPointField(
        ("deep_mesh",), "Bias phase, rad: pi V_b / V_pi (Eq. 5)."),
    "eo_tia_gain_ohm": OperatingPointField(
        ("deep_mesh",), "Transimpedance gain G of the activation's amplifier."),
    "eo_responsivity_a_per_w": OperatingPointField(
        ("deep_mesh",), "Photodiode responsivity R."),
    "eo_v_pi": OperatingPointField(
        ("deep_mesh",), "Half-wave voltage of the activation's phase modulator."),
    "eo_v_bias": OperatingPointField(
        ("deep_mesh",), "Static bias voltage V_b on that modulator."),
    "eo_bandwidth_hz": OperatingPointField(
        ("deep_mesh",),
        "Electrical bandwidth of the photodiode-amplifier loop. The activation must "
        "answer within one input symbol, so this, not the readout's integration "
        "time, sets the noise on the light-written phase."),
}

#: Relative agreement required between the derived activation constants and the raw
#: device values they are computed from.
_EO_CONSISTENCY_RTOL = 1e-9


def _check_eo_consistency(operating_point):
    """Raise unless eo_g_phi and eo_phi_b follow from the raw device values."""
    op = operating_point
    g = (np.pi * op["eo_alpha"] * op["eo_tia_gain_ohm"] * op["eo_responsivity_a_per_w"]
         / op["eo_v_pi"])
    b = np.pi * op["eo_v_bias"] / op["eo_v_pi"]
    for name, derived, raw in (("eo_g_phi", op["eo_g_phi"], g),
                               ("eo_phi_b", op["eo_phi_b"], b)):
        if not np.isclose(derived, raw, rtol=_EO_CONSISTENCY_RTOL, atol=0.0):
            raise ValueError(
                f"operating_point {name}={derived!r} does not follow from the raw device "
                f"values ({raw!r}; Williamson et al. Eqs. 5 and 7). The as-built model "
                "perturbs the raw values, so the two must describe one device."
            )


def _check_operating_point(operating_point, model_type):
    """Reject an operating point the manifest does not recognise or does not cover."""
    unknown = sorted(set(operating_point) - set(OPERATING_POINT))
    if unknown:
        raise ValueError(
            f"operating_point has unrecognised key(s) {unknown}. Add them to "
            "photonn.export.OPERATING_POINT (and to the MATLAB reader) rather "
            "than writing a scalar no reader knows to look for."
        )
    missing = sorted(
        key for key, spec in OPERATING_POINT.items()
        if model_type in spec.required_for and key not in operating_point
    )
    if missing:
        raise ValueError(
            f"operating_point is missing {missing} for model_type={model_type!r}. "
            "Every one of these is read downstream; absent, MATLAB substitutes a "
            "default and produces a plausible wrong answer."
        )
    if model_type == "deep_mesh":
        _check_eo_consistency(operating_point)


def _as_str(value):
    """Decode an HDF5 attribute that may come back as bytes into ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


#: Mesh parameter datasets, in the order they are written and checked.
_MESH_DATASETS = ("phase_theta", "phase_phi", "sigma", "out_phase")


def _region_array(regions) -> np.ndarray:
    """Detector patches as ``i4[n_classes, 4]``: ``(y0, y1, x0, x1)`` per class.

    Accepts either :class:`photonn.detect.DetectorRegion` objects or plain
    4-sequences, so a caller can pass ``default_regions(...)`` straight through.
    Half-open in y1/x1, matching the Python slices; the MATLAB reader converts to
    its own 1-based inclusive form once, on the way in.
    """
    rows = []
    for reg in regions:
        if hasattr(reg, "y0"):
            rows.append((reg.y0, reg.y1, reg.x0, reg.x1))
        else:
            y0, y1, x0, x1 = reg
            rows.append((y0, y1, x0, x1))
    out = np.asarray(rows, dtype="i4")
    if out.ndim != 2 or out.shape[1] != 4:
        raise ValueError(f"detector_regions must be [n_classes, 4]; got {out.shape}.")
    return out


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


def _deep_mesh_arrays(parameters):
    """Coerce and cross-check the per-layer arrays of a ``deep_mesh``.

    ``phase_theta``/``phase_phi`` are ``[n_layers, 2 * n_mzi]`` (each row the
    :data:`MESH_ORDER` concatenation one ``mesh`` file carries), ``sigma`` is
    ``[n_layers, n_modes]`` and ``out_phase`` ``[n_layers, 2, n_modes]``. Every sigma
    must lie in ``[0, 1]``: those ahead of an activation are passive by construction,
    and the last is passivized at export.
    """
    for key in _MESH_DATASETS:
        if key not in parameters:
            raise ValueError(f"deep_mesh parameters are missing required key {key!r}.")
    theta = np.asarray(parameters["phase_theta"], dtype="f8")
    phi = np.asarray(parameters["phase_phi"], dtype="f8")
    sigma = np.asarray(parameters["sigma"], dtype="f8")
    out_phase = np.asarray(parameters["out_phase"], dtype="f8")
    if out_phase.ndim != 3 or out_phase.shape[1] != 2:
        raise ValueError(
            f"'out_phase' must be [n_layers, 2, n_modes]; got {out_phase.shape}.")
    n_layers, _, n_modes = out_phase.shape
    if n_layers < 2:
        raise ValueError(f"a deep_mesh has at least two layers; got {n_layers}.")
    n_mzi = n_modes * (n_modes - 1) // 2
    for name, arr in (("phase_theta", theta), ("phase_phi", phi)):
        if arr.shape != (n_layers, 2 * n_mzi):
            raise ValueError(
                f"{name!r} must be [n_layers, 2 * n_mzi]={(n_layers, 2 * n_mzi)}; "
                f"got {arr.shape}.")
    if sigma.shape != (n_layers, n_modes):
        raise ValueError(f"'sigma' must be [n_layers, n_modes]={(n_layers, n_modes)}; "
                         f"got {sigma.shape}.")
    if sigma.min() < 0.0 or sigma.max() > 1.0:
        raise ValueError(
            f"deep_mesh sigma must lie in [0, 1] (passive); got {sigma.min():.4g} .. "
            f"{sigma.max():.4g}. A sigma above one ahead of an activation is gain no "
            "chip has, and the last layer should be passivized with mzi.passivize.")
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
    test_acc=None,
    test_inputs=None,
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
    test_inputs : array_like of complex, optional
        The encoded, unit-norm input field per test image, ``[n, n_modes]``, written
        as ``/test_set/inputs_re`` and ``inputs_im``. Required for ``deep_mesh``;
        allowed for ``mesh``, whose readers fall back to the magnitude map in
        ``images`` when it is absent (every file before 0.4.0).
    description : str, optional
        Free-text note stored at the file root.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"model_type must be one of {MODEL_TYPES}; got {model_type!r}."
        )
    _check_operating_point(operating_point, model_type)
    for key in ("grid_size", "physical_extent_m", "n_layers", "layer_separations_m"):
        if key not in geometry:
            raise ValueError(f"geometry is missing required key {key!r}.")
    if model_type == "deep_mesh" and test_inputs is None:
        raise ValueError(
            "a deep_mesh handoff must carry test_inputs: its encoded input is complex, "
            "and the image map in test_images cannot describe it.")
    if model_type == "d2nn" and "detector_regions" not in geometry:
        raise ValueError(
            "geometry is missing 'detector_regions' for a d2nn handoff. Pass "
            "photonn.detect.default_regions(n, n_classes); the as-built model "
            "must read the layout rather than re-deriving it from constants "
            "typed on the other side of the seam."
        )

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["created"] = datetime.now(timezone.utc).isoformat()
        f.attrs["description"] = description
        # A real attribute, not a token inside free text. The accuracy is read
        # back by two exporters, and both used to dig it out of the description
        # with byte-identical hand-rolled parsers. photonn.handoff still falls
        # back to that parse, for the files written before this line existed.
        if test_acc is not None:
            f.attrs["test_acc"] = float(test_acc)

        geo = f.create_group("geometry")
        geo.attrs["grid_size"] = int(geometry["grid_size"])
        geo.attrs["physical_extent_m"] = float(geometry["physical_extent_m"])
        geo.attrs["n_layers"] = int(geometry["n_layers"])
        geo.create_dataset(
            "layer_separations_m",
            data=np.asarray(geometry["layer_separations_m"], dtype="f8"),
        )
        if "detector_regions" in geometry:
            geo.create_dataset(
                "detector_regions", data=_region_array(geometry["detector_regions"])
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
        elif model_type == "deep_mesh":
            theta, phi, sigma, out_phase = _deep_mesh_arrays(parameters)
            n_layers, _, n_modes = out_phase.shape
            p.attrs["n_modes"] = int(n_modes)
            p.attrs["n_mzi_per_mesh"] = int(n_modes * (n_modes - 1) // 2)
            p.attrs["n_layers"] = int(n_layers)
            p.attrs["mesh_order"] = MESH_ORDER
            p.attrs["topology"] = MESH_TOPOLOGY
            p.attrs["activation"] = "williamson2020_eo"
            p.create_dataset("phase_theta", data=theta)
            p.create_dataset("phase_phi", data=phi)
            p.create_dataset("sigma", data=sigma)
            p.create_dataset("out_phase", data=out_phase)
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
        if test_inputs is not None:
            inputs = np.asarray(test_inputs, dtype=complex)
            if inputs.ndim != 2 or inputs.shape[0] != len(test_labels):
                raise ValueError(
                    f"test_inputs must be [n_test, n_modes]; got {inputs.shape} for "
                    f"{len(test_labels)} labels.")
            ts.create_dataset("inputs_re", data=inputs.real.astype("f8"))
            ts.create_dataset("inputs_im", data=inputs.imag.astype("f8"))


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

        op_attrs = f["operating_point"].attrs
        if "wavelength_m" not in op_attrs:
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
            # Files written before 0.3.0 carry no layout and readers derive it,
            # which is what they did unconditionally before. From 0.3.0 the file
            # is the authority and its absence is a real gap.
            if version >= "0.3.0" and "detector_regions" not in geo:
                raise ValueError(
                    "Missing dataset '/geometry/detector_regions'. From schema "
                    "0.3.0 the detector layout crosses the seam as data; see "
                    "photonn.detect.default_regions."
                )
        elif model_type == "deep_mesh":
            required = _MESH_DATASETS
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
        if model_type == "deep_mesh":
            _deep_mesh_arrays({k: params[k][...] for k in _MESH_DATASETS})
            for attr in ("n_modes", "n_mzi_per_mesh", "n_layers", "mesh_order",
                         "topology", "activation"):
                if attr not in params.attrs:
                    raise ValueError(f"Missing attribute '/parameters.{attr}'.")

        # The same manifest the writer enforces, checked against the bytes. A file
        # can reach here without having gone through write_handoff (hand-edited,
        # or produced by an older exporter), and the readers do not care which.
        missing = sorted(
            key for key, spec in OPERATING_POINT.items()
            if model_type in spec.required_for and key not in op_attrs
        )
        if missing:
            raise ValueError(
                f"Missing '/operating_point' attribute(s) {missing} for "
                f"model_type={model_type!r}. See photonn.export.OPERATING_POINT."
            )

        if model_type == "deep_mesh":
            _check_eo_consistency({k: float(op_attrs[k]) for k in op_attrs})

        test_set = f["test_set"]
        required_ts = ("images", "labels")
        if model_type == "deep_mesh":
            required_ts += ("inputs_re", "inputs_im")
        for dset in required_ts:
            if dset not in test_set:
                raise ValueError(f"Missing dataset '/test_set/{dset}'.")
