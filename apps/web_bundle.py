"""Encoding shared by every browser weight bundle.

Two exporters write bundles that ``apps/web/d2nn.js:buildNet`` decodes:
:mod:`apps.export_d2nn_web` (a trained model with a full Phase-2 handoff) and
:mod:`apps.export_sweep_web` (a configuration lifted out of the optics sweep).
Both have to agree with that decoder byte for byte, so the encoding lives here
once rather than being written out twice and drifting.

The 8-bit path is not a size shortcut. ``docs/tolerance_d2nn.md`` measures this
design as holding accuracy down to **3-bit** phase control, and 8 bits is what a
real SLM offers, so the quantised model is the *more* faithful one. That it is
also four times smaller is what lets two trained models share a page.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np

#: Bit depths ``d2nn.js`` can decode. 32 is little-endian float32 radians (no
#: ``masks_bits`` key); 8 is one uint8 code per phase, keyed by ``masks_bits: 8``.
DECODABLE_BITS = (8, 32)


def wrap_phase(masks: np.ndarray) -> np.ndarray:
    """Wrap phases to [-pi, pi) in float64.

    ``exp(i*phi)`` is invariant to a 2*pi wrap, so this costs nothing physically
    and buys precision in both encodings: it keeps the float32 mantissa on the
    part that matters (the raw parameters range over about +-9.2 rad), and it is
    the interval the 8-bit codes are defined on.
    """
    return (np.asarray(masks, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def encode_masks(masks: np.ndarray, bits: int = 32) -> tuple[str, float]:
    """Encode phase masks for the browser -> (base64, max encoding error in rad).

    The error is returned rather than assumed so a caller can print it beside the
    model: for the 8-bit path it is the quantisation step, which is the number
    that has to be read against the tolerance budget.
    """
    wrapped = wrap_phase(masks)

    if bits == 32:
        payload = np.ascontiguousarray(wrapped, dtype="<f4")
        err = float(np.abs(payload.astype(np.float64) - wrapped).max())
        return base64.b64encode(payload.tobytes()).decode("ascii"), err

    if bits != 8:
        raise ValueError(f"masks_bits={bits} is not decoded by d2nn.js; use one of {DECODABLE_BITS}.")

    step = 2 * np.pi / 256
    codes = np.clip(np.rint((wrapped + np.pi) / step), 0, 255).astype(np.uint8)
    recovered = codes.astype(np.float64) * step - np.pi
    return (base64.b64encode(codes.tobytes()).decode("ascii"),
            float(np.abs(recovered - wrapped).max()))


def b64(arr: np.ndarray, dtype: str) -> str:
    """Base64 of ``arr`` as little-endian ``dtype`` (decoded by a typed array in JS)."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode("ascii")


def read_bundle(path) -> dict:
    """Parse a generated bundle's payload back out of its IIFE.

    Bundles are JavaScript because that is what the page loads, but the payload
    is plain JSON inside it -- which is how one exporter can copy another's
    gallery, and how the tests read a bundle without running Node.
    """
    text = Path(path).read_text(encoding="utf-8")
    body = re.search(r"var W = (\{.*?\});\n", text, re.S)
    if body is None:
        raise SystemExit(f"{path} does not look like a generated weights bundle.")
    return json.loads(body.group(1))


def js_global(out_path) -> str:
    """The ``window`` name a bundle publishes itself under, from its filename.

    ``d2nn_weights.js`` -> ``D2NN_WEIGHTS``, ``d2nn_deep_weights.js`` ->
    ``D2NN_DEEP_WEIGHTS``. Deriving it means a second model needs a filename and
    nothing else, and that two bundles can never collide on one page.
    """
    name = Path(out_path).stem.upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
        raise SystemExit(f"{out_path} does not give a usable JS identifier ({name!r}).")
    return name
