"""Reading the design -> as-built handoff back, from Python.

:mod:`photonn.export` writes the file and MATLAB reads it. Python reads it too --
four separate modules did, each opening ``h5py`` itself and re-stating the parts
of the schema it needed by literal string:
:mod:`apps.export_d2nn_web`, :mod:`apps.export_analogy_web`,
:mod:`apps.export_mesh_web`, and the mesh sufficiency test. Two of them carried
byte-identical copies of a parser that dug the trained accuracy out of the
description's free text.

This module is the one reader. It does not weaken the one-directional rule in
CLAUDE.md: that rule is about MATLAB never writing back into the Python pipeline,
and nothing here writes anything.

Reading is split by cost, because it has to be. ``exports/d2nn_phase2.h5`` is
131 MB and :func:`read_handoff` is often wanted for four scalars, so it loads the
metadata groups only; the masks and the frozen test set come from
:func:`read_parameters` and :func:`read_test_set` when they are actually needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

from photonn.export import (MODEL_TYPES, OPERATING_POINT, SUPPORTED_SCHEMAS,
                            _as_str)


@dataclass(frozen=True)
class Handoff:
    """The metadata half of a handoff file: everything except the big arrays."""

    path: str
    schema_version: str
    model_type: str
    description: str
    geometry: dict
    operating_point: dict

    # -- geometry, named ------------------------------------------------------
    @property
    def n(self) -> int:
        """Grid size in pixels."""
        return int(self.geometry["grid_size"])

    @property
    def n_layers(self) -> int:
        return int(self.geometry["n_layers"])

    @property
    def separations(self) -> np.ndarray:
        return self.geometry["layer_separations_m"]

    @property
    def separation(self) -> float:
        """The single plane separation, for the uniform stacks every model here uses.

        Raises if the file is not uniform rather than quietly returning the first
        gap: both the browser forward pass and the correspondence figure build one
        transfer function on this assumption, and each used to check it separately.
        """
        seps = self.separations
        if not np.allclose(seps, seps[0]):
            raise ValueError(
                f"{self.path}: non-uniform layer separations {seps}; callers that "
                "build a single transfer function assume one z."
            )
        return float(seps[0])

    @property
    def regions(self):
        """Detector patches as :class:`photonn.detect.DetectorRegion` objects.

        Read from ``/geometry/detector_regions`` when the file carries it
        (schema 0.3.0 on). Older files predate the layout crossing the seam, so
        it is derived from the same defaults every reader used to derive it from
        -- explicitly, and only here, rather than in each of them.
        """
        from photonn.detect import DetectorRegion, default_regions

        raw = self.geometry.get("detector_regions")
        if raw is None:
            return default_regions(self.n, 10)
        return [DetectorRegion(int(y0), int(y1), int(x0), int(x1), label=i)
                for i, (y0, y1, x0, x1) in enumerate(raw)]

    # -- operating point ------------------------------------------------------
    def op(self, key: str) -> float:
        """One operating-point constant, by manifest name."""
        if key not in OPERATING_POINT:
            raise KeyError(
                f"{key!r} is not an operating-point constant. Known: "
                f"{sorted(OPERATING_POINT)}"
            )
        if key not in self.operating_point:
            raise KeyError(
                f"{self.path} carries no '/operating_point.{key}', which "
                f"{self.model_type!r} handoffs require. "
                f"{OPERATING_POINT[key].doc} Re-export the model."
            )
        return float(self.operating_point[key])

    @property
    def test_acc(self) -> float:
        """Test accuracy of the exported model.

        Read from the root attribute when present. Files written before that
        attribute existed -- including the two 131 MB D2NN exports still on disk
        -- carry it only inside the free-text description, so that is parsed as a
        fallback. Two modules used to hold byte-identical copies of this parser.
        """
        if "test_acc" in self._root_attrs:
            return float(self._root_attrs["test_acc"])
        for token in str(self.description).split("|"):
            key, _, value = token.strip().partition("=")
            if key == "test_acc":
                return float(value)
        raise KeyError(
            f"{self.path} states no test_acc, as an attribute or in its description."
        )

    _root_attrs: dict = None


def read_handoff(path) -> Handoff:
    """Read a handoff's metadata, checking it against the schema on the way in.

    Cheap: touches the attribute groups and ``layer_separations_m`` only.
    """
    path = str(path)
    with h5py.File(path, "r") as f:
        root = dict(f.attrs)
        version = _as_str(root.get("schema_version", ""))
        if version not in SUPPORTED_SCHEMAS:
            raise ValueError(
                f"{path} is schema {version!r}; this reader supports "
                f"{SUPPORTED_SCHEMAS!r}."
            )
        model_type = _as_str(f["parameters"].attrs["model_type"])
        if model_type not in MODEL_TYPES:
            raise ValueError(f"{path}: unknown model_type {model_type!r}.")

        geometry = dict(f["geometry"].attrs)
        geometry["layer_separations_m"] = f["geometry"]["layer_separations_m"][...]
        if "detector_regions" in f["geometry"]:
            geometry["detector_regions"] = f["geometry"]["detector_regions"][...]
        operating_point = dict(f["operating_point"].attrs)

    missing = sorted(
        key for key, spec in OPERATING_POINT.items()
        if model_type in spec.required_for and key not in operating_point
    )
    if missing:
        raise ValueError(
            f"{path} is missing operating-point constant(s) {missing} required by "
            f"model_type={model_type!r}. A reader that defaults these produces a "
            "plausible wrong answer, so this refuses instead."
        )

    return Handoff(
        path=path,
        schema_version=version,
        model_type=model_type,
        description=_as_str(root.get("description", "")),
        geometry=geometry,
        operating_point=operating_point,
        _root_attrs=root,
    )


def read_parameters(path) -> dict:
    """The trained parameter arrays: masks for a d2nn, the four arrays for a mesh."""
    with h5py.File(str(path), "r") as f:
        p = f["parameters"]
        out = {k: p[k][...] for k in p.keys()}
        out.update({k: p.attrs[k] for k in p.attrs})
    return out


def read_test_set(path):
    """``(images, labels)`` of the frozen test set."""
    with h5py.File(str(path), "r") as f:
        return f["test_set/images"][...], f["test_set/labels"][...]
