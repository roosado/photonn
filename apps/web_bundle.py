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
#: ``masks_bits`` key); 8 is one uint8 code per phase; 4 packs two codes per byte,
#: high nibble first. The quantised depths are keyed by ``masks_bits``.
DECODABLE_BITS = (4, 8, 32)


def wrap_phase(masks: np.ndarray) -> np.ndarray:
    """Wrap phases to [-pi, pi) in float64.

    ``exp(i*phi)`` is invariant to a 2*pi wrap, so this costs nothing physically
    and buys precision in both encodings: it keeps the float32 mantissa on the
    part that matters (the raw parameters range over about +-9.2 rad), and it is
    the interval the 8-bit codes are defined on.
    """
    return (np.asarray(masks, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def _codes(wrapped: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    """Phase codes and the code step, for a quantised bit depth."""
    levels = 1 << bits
    step = 2 * np.pi / levels
    codes = np.clip(np.rint((wrapped + np.pi) / step), 0, levels - 1).astype(np.uint8)
    return codes, step


def quantise_phase(masks: np.ndarray, bits: int = 32) -> np.ndarray:
    """The phases the browser will actually apply, as float64 radians.

    This exists so a caller can score *the model it is shipping* rather than the
    float model it started from. The exporter runs the torch cross-check through
    this, which is what keeps a bundle's stated accuracy honest and keeps
    ``tests/fixtures/d2nn_reference.json`` describing the same network the page
    runs.
    """
    wrapped = wrap_phase(masks)
    if bits == 32:
        return np.ascontiguousarray(wrapped, dtype="<f4").astype(np.float64)
    if bits not in DECODABLE_BITS:
        raise ValueError(f"masks_bits={bits} is not decoded by d2nn.js; use one of {DECODABLE_BITS}.")
    codes, step = _codes(wrapped, bits)
    return codes.astype(np.float64) * step - np.pi


def encode_masks(masks: np.ndarray, bits: int = 32) -> tuple[str, float]:
    """Encode phase masks for the browser -> (base64, max encoding error in rad).

    The error is returned rather than assumed so a caller can print it beside the
    model: for a quantised path it is the quantisation step, which is the number
    that has to be read against the tolerance budget.
    """
    wrapped = wrap_phase(masks)

    if bits == 32:
        payload = np.ascontiguousarray(wrapped, dtype="<f4")
        err = float(np.abs(payload.astype(np.float64) - wrapped).max())
        return base64.b64encode(payload.tobytes()).decode("ascii"), err

    if bits not in DECODABLE_BITS:
        raise ValueError(f"masks_bits={bits} is not decoded by d2nn.js; use one of {DECODABLE_BITS}.")

    codes, step = _codes(wrapped, bits)
    err = float(np.abs((codes.astype(np.float64) * step - np.pi) - wrapped).max())

    if bits == 8:
        payload = codes.tobytes()
    else:
        # Two codes per byte, high nibble first -- the order d2nn.js unpacks. Every
        # mask stack is n_layers * n * n with n even, so the count is never odd.
        flat = codes.ravel()
        if flat.size % 2:
            raise ValueError(f"4-bit packing needs an even phase count, got {flat.size}.")
        payload = ((flat[0::2] << 4) | flat[1::2]).astype(np.uint8).tobytes()

    return base64.b64encode(payload).decode("ascii"), err


def decode_masks(masks_b64: str, bits: int, shape: tuple[int, ...]) -> np.ndarray:
    """Inverse of :func:`encode_masks` -> phase masks in radians, shaped ``shape``.

    This is the reference implementation of what ``d2nn.js:buildNet`` does to a
    bundle, and exists so a test can rebuild the browser's model without a
    second, drifting copy of the unpacking rules.
    """
    raw = np.frombuffer(base64.b64decode(masks_b64), dtype=np.uint8)

    if bits == 32:
        return np.frombuffer(raw.tobytes(), dtype="<f4").astype(np.float64).reshape(shape)
    if bits not in DECODABLE_BITS:
        raise ValueError(f"masks_bits={bits} is not decoded by d2nn.js; use one of {DECODABLE_BITS}.")

    if bits == 8:
        codes = raw
    else:
        codes = np.empty(raw.size * 2, dtype=np.uint8)
        codes[0::2] = raw >> 4
        codes[1::2] = raw & 15

    step = 2 * np.pi / (1 << bits)
    return (codes.astype(np.float64) * step - np.pi).reshape(shape)


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
