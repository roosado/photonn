"""The phase encoding a bundle ships in, and what it costs.

``apps/web_bundle.py`` and ``apps/web/d2nn.js`` have to agree byte for byte about
how a phase mask is packed, and there is no shared runtime to enforce it -- one
is Python, the other runs in the visitor's browser. These tests pin the Python
side against its own inverse and against the arithmetic the JS decoder performs,
so a change to one that is not mirrored in the other fails here rather than on
the deployed page.

The bit depth is not free choice: it sets how large the self-contained pages are.
At 128x128 a 56-mask stack is 917,504 phases, which is 1,195 KB of base64 at
8 bits and 597 KB at 4. See ``docs/tolerance_d2nn.md`` for why 4 bits is inside
what the hardware model already assumes (the budget holds this design to 3-bit
phase control).
"""
from __future__ import annotations

import base64

import numpy as np
import pytest

from apps.web_bundle import DECODABLE_BITS, decode_masks, encode_masks, quantise_phase, wrap_phase


@pytest.fixture
def masks():
    """A small stack with a deterministic spread of phases, including the wrap edges."""
    rng = np.random.default_rng(20260810)
    m = rng.uniform(-9.2, 9.2, size=(3, 8, 8))       # raw parameters run well past +-pi
    m.flat[0], m.flat[1], m.flat[2] = -np.pi, 0.0, np.pi - 1e-9
    return m


@pytest.mark.parametrize("bits", DECODABLE_BITS)
def test_encode_decode_round_trips(masks, bits):
    """decode(encode(x)) is the phase the browser applies, to the last bit."""
    b64, _ = encode_masks(masks, bits)
    got = decode_masks(b64, bits, masks.shape)
    assert got.shape == masks.shape
    np.testing.assert_allclose(got, quantise_phase(masks, bits), rtol=0, atol=0)


@pytest.mark.parametrize("bits", DECODABLE_BITS)
def test_encoding_error_is_reported_and_real(masks, bits):
    """The returned error bounds the actual deviation -- callers print it as fact."""
    b64, err = encode_masks(masks, bits)
    got = decode_masks(b64, bits, masks.shape)
    worst = float(np.abs(got - wrap_phase(masks)).max())
    assert worst <= err + 1e-12, f"reported {err:.6f} rad but measured {worst:.6f}"
    # Half a code step, plus room for the wrap edge landing on a boundary.
    if bits != 32:
        assert err <= 2 * np.pi / (1 << bits)


@pytest.mark.parametrize("bits,per_phase", [(4, 0.5), (8, 1.0), (32, 4.0)])
def test_payload_is_the_size_the_page_budget_assumes(masks, bits, per_phase):
    """Bytes per phase is the whole reason a depth is chosen; assert it directly."""
    b64, _ = encode_masks(masks, bits)
    assert len(base64.b64decode(b64)) == pytest.approx(masks.size * per_phase)


def test_four_bit_packs_high_nibble_first(masks):
    """The nibble order d2nn.js unpacks. Reversing it silently scrambles the masks."""
    b64, _ = encode_masks(masks, 4)
    raw = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)

    step = 2 * np.pi / 16
    codes = np.clip(np.rint((wrap_phase(masks).ravel() + np.pi) / step), 0, 15).astype(np.uint8)
    np.testing.assert_array_equal(raw >> 4, codes[0::2])
    np.testing.assert_array_equal(raw & 15, codes[1::2])


def test_codes_stay_inside_the_lookup_table(masks):
    """Every code must index the 2^bits table d2nn.js builds; an out-of-range one
    reads undefined and poisons the field with NaN rather than failing loudly."""
    for bits in (4, 8):
        raw = np.frombuffer(base64.b64decode(encode_masks(masks, bits)[0]), dtype=np.uint8)
        codes = raw if bits == 8 else np.concatenate([raw >> 4, raw & 15])
        assert codes.max() < (1 << bits)


def test_an_undecodable_depth_is_refused(masks):
    """Better to fail in the exporter than to ship a bundle no browser can read."""
    with pytest.raises(ValueError, match="not decoded by d2nn.js"):
        encode_masks(masks, 6)
    with pytest.raises(ValueError, match="not decoded by d2nn.js"):
        decode_masks("AAAA", 6, (1, 1, 2))


def test_quantise_phase_is_what_encode_masks_encodes(masks):
    """The exporter scores the model through quantise_phase and ships encode_masks.
    If these two ever disagree, every stated accuracy describes a different network
    than the one the page runs."""
    for bits in DECODABLE_BITS:
        b64, _ = encode_masks(masks, bits)
        np.testing.assert_allclose(
            decode_masks(b64, bits, masks.shape), quantise_phase(masks, bits), rtol=0, atol=0)
