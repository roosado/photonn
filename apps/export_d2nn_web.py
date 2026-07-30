"""Export the trained D2NN for the browser demo, plus its cross-check fixture.

Bridges the trained model to ``apps/web/d2nn.js``. Reads the Phase-2 handoff
(``exports/d2nn_phase2.h5``) and the torch checkpoint, and writes:

- ``apps/web/d2nn_weights.js`` -- the browser bundle: phase masks (base64
  float32), geometry, operating point, detector regions, and a small gallery of
  frozen test digits. **This file is committed**: ``*.h5`` and ``*.pt`` are
  gitignored, so it is the only in-repo copy of the trained model and the only
  way ``apps.build_site`` can rebuild the page from a fresh clone.
- ``tests/fixtures/d2nn_reference.json`` -- the authoritative torch logits for a
  frozen set of digits, consumed by ``tests/test_d2nn_crosscheck.py``.

Every value comes from the handoff file; nothing about the operating point is
hardcoded here. Run from the repo root in the project venv::

    python -m apps.export_d2nn_web

The frozen ``/test_set`` is stored at grid resolution (the 128x128 embedded
canvas), not as 28x28 digits, so the gallery re-derives the original MNIST
digits with :func:`photonn.train.load_dataset`. That reproduction is *asserted*
against the handoff before anything is written -- a changed MNIST cache would
otherwise silently ship a gallery whose labels no longer match the frozen set.
"""
from __future__ import annotations

import base64
import json
import os

import h5py
import numpy as np
import torch

from photonn.detect import default_regions
from photonn.models import D2NN
from photonn.train import embed_input, load_dataset

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5_PATH = os.path.join(_REPO, "exports", "d2nn_phase2.h5")
PT_PATH = os.path.join(_REPO, "exports", "d2nn_phase2.pt")
WEIGHTS_JS = os.path.join(_REPO, "apps", "web", "d2nn_weights.js")
FIXTURE_JSON = os.path.join(_REPO, "tests", "fixtures", "d2nn_reference.json")

#: Digits shipped in the browser gallery (all ten classes, plus known failures).
N_GALLERY_CORRECT = 10   # one per class the model gets right -- shows it works
N_GALLERY_WRONG = 6      # ones it gets wrong -- shown honestly, not hidden
#: Digits in the JS<->torch cross-check fixture.
N_FIXTURE = 24
#: Digits carrying a full reference canvas, to validate the JS bilinear resize.
N_RESIZE_CASES = 4
#: Canvases checked when verifying the MNIST reproduction (all labels are checked).
N_VERIFY_CANVAS = 256


def _b64(arr: np.ndarray, dtype: str) -> str:
    """Base64 of ``arr`` as little-endian ``dtype`` (decoded by a typed array in JS)."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode("ascii")


def load_handoff():
    """Read geometry, operating point, masks and the frozen test set from the h5."""
    with h5py.File(H5_PATH, "r") as f:
        geo, op = dict(f["geometry"].attrs), dict(f["operating_point"].attrs)
        seps = f["geometry"]["layer_separations_m"][...]
        masks = f["parameters/phase_masks"][...]
        labels = f["test_set/labels"][...]
        canvases = f["test_set/images"][...]

    n = int(geo["grid_size"])
    n_layers = int(geo["n_layers"])
    dx = float(op["pixel_pitch_m"])

    # The D2NN propagates one uniform distance between every pair of planes; the
    # browser forward pass builds a single transfer function on that assumption.
    if not np.allclose(seps, seps[0]):
        raise ValueError(f"non-uniform layer separations {seps}; the browser demo assumes one z.")
    if masks.shape != (n_layers, n, n):
        raise ValueError(f"phase_masks {masks.shape} does not match n_layers={n_layers}, n={n}.")
    if not np.isclose(float(geo["physical_extent_m"]), n * dx):
        raise ValueError("physical_extent_m disagrees with grid_size * pixel_pitch_m.")
    if int(op["encoding_code"]) != 2:
        raise ValueError(f"expected encoding_code=2 ('both'); got {op['encoding_code']}.")

    return dict(
        n=n, n_layers=n_layers, dx=dx,
        wavelength=float(op["wavelength_m"]),
        separation=float(seps[0]),
        readout_gain=float(op["readout_gain"]),
        phase_scale=float(op["phase_scale_rad"]),
        input_frac=float(op["input_frac"]),
        masks=masks, labels=labels, canvases=canvases,
    )


def recover_mnist(hand):
    """Re-derive the 28x28 digits behind the frozen test set, and prove it.

    ``/test_set/images`` holds the 128x128 embedded canvas, not the source digit,
    so the gallery needs the originals back. ``load_dataset`` is deterministic
    (fixed subset seed), so it reproduces the exact frozen subset -- which we
    verify against the handoff rather than trust.
    """
    n, labels, canvases = hand["n"], hand["labels"], hand["canvases"]
    ds = load_dataset("mnist", subset=len(labels), split="test")

    if not np.array_equal(ds.labels.astype("i4"), labels):
        raise RuntimeError(
            "load_dataset did not reproduce the frozen test labels -- the MNIST cache or "
            "the subset seed changed. Refusing to write a mismatched gallery."
        )
    sample = np.linspace(0, len(labels) - 1, N_VERIFY_CANVAS, dtype=int)
    redone = embed_input(
        torch.as_tensor(ds.images[sample], dtype=torch.float32),
        n=n, input_frac=hand["input_frac"],
    ).numpy()
    err = float(np.abs(redone - canvases[sample]).max())
    if err != 0.0:
        raise RuntimeError(f"re-embedded canvases differ from the handoff (max {err:.3e}).")
    print(f"  verified: {len(labels)} labels and {N_VERIFY_CANVAS} canvases reproduce exactly")
    return ds.images


def load_model(hand):
    """Rebuild the trained D2NN from the handoff geometry + the torch checkpoint."""
    model = D2NN(
        n=hand["n"], n_layers=hand["n_layers"], dx=hand["dx"],
        wavelength=hand["wavelength"], separation=hand["separation"],
        readout_gain=hand["readout_gain"],
    )
    ckpt = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # The checkpoint must be the same model the handoff describes.
    got = np.stack([m.phi.detach().numpy() for m in model.masks]).astype("f8")
    if not np.array_equal(got, hand["masks"]):
        raise RuntimeError("checkpoint phase masks differ from the handoff's phase_masks.")
    return model


@torch.no_grad()
def torch_logits(model, canvases, phase_scale, batch=100):
    """Run the trained model on the frozen canvases -> (logits, predictions).

    Encodes exactly as :func:`photonn.train.encode_input` with ``scheme="both"``,
    but from the stored canvas rather than the raw digit, so this is the ground
    truth the JavaScript must reproduce.
    """
    out = []
    for i in range(0, len(canvases), batch):
        c = torch.as_tensor(canvases[i:i + batch], dtype=torch.float32)
        field = torch.polar(c, c * phase_scale)
        field = field / field.abs().pow(2).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12).sqrt()
        out.append(model(field.to(torch.complex64)))
    logits = torch.cat(out).numpy()
    return logits, logits.argmax(axis=1)


def pick_gallery(labels, preds):
    """Choose gallery digits: one correct per class, then a few honest failures."""
    correct = [int(np.flatnonzero((labels == c) & (preds == labels))[0]) for c in range(10)]
    wrong = np.flatnonzero(preds != labels)
    # Spread the failures over distinct true classes so the gallery isn't all one digit.
    picked, seen = [], set()
    for i in wrong:
        if labels[i] not in seen:
            picked.append(int(i))
            seen.add(labels[i])
        if len(picked) == N_GALLERY_WRONG:
            break
    return correct + picked


def write_weights_js(hand, digits28, regions, gallery_idx, labels, path=WEIGHTS_JS):
    """Write the browser weight bundle."""
    # exp(i*phi) is invariant to a 2*pi wrap, so wrapping in float64 before the
    # float32 cast costs nothing and keeps the mantissa on the part that matters
    # (the raw parameters range over about +-9.2 rad).
    wrapped = (hand["masks"] + np.pi) % (2 * np.pi) - np.pi
    gallery = np.round(digits28[gallery_idx] * 255.0).astype(np.uint8)

    payload = {
        "n": hand["n"],
        "dx": hand["dx"],
        "wavelength": hand["wavelength"],
        "separation": hand["separation"],
        "n_layers": hand["n_layers"],
        "readout_gain": hand["readout_gain"],
        "phase_scale": hand["phase_scale"],
        "input_frac": hand["input_frac"],
        "regions": [[r.y0, r.y1, r.x0, r.x1] for r in regions],
        "masks_b64": _b64(wrapped, "<f4"),
        "gallery_b64": _b64(gallery, "u1"),
        "gallery_labels": [int(labels[i]) for i in gallery_idx],
        "gallery_size": 28,
    }
    body = ",\n    ".join(f'"{k}": {json.dumps(v)}' for k, v in payload.items())
    js = f"""/*
 * d2nn_weights.js -- the trained Phase-2 diffractive network, for the browser.
 *
 * GENERATED by apps/export_d2nn_web.py from exports/d2nn_phase2.h5 -- do not edit
 * by hand; re-run the exporter instead. This file is committed because the h5 and
 * pt exports are gitignored, so it is the repo's only copy of the trained model.
 *
 * masks_b64   : {hand['n_layers']} x {hand['n']} x {hand['n']} float32 phase (radians, wrapped to [-pi, pi)),
 *               little-endian, applied as E * exp(i * phi).
 * gallery_b64 : {len(gallery_idx)} x 28 x 28 uint8 MNIST digits from the frozen test set.
 * regions     : ten [y0, y1, x0, x1] detector boxes, class order, from
 *               photonn.detect.default_regions.
 * All lengths in metres.
 */
(function () {{
  "use strict";
  var W = {{
    {body}
  }};
  if (typeof module !== "undefined" && module.exports) module.exports = W;
  if (typeof window !== "undefined") window.D2NN_WEIGHTS = W;
}})();
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(js)
    return len(js)


def write_fixture(hand, digits28, canvases, logits, preds, labels, path=FIXTURE_JSON):
    """Write the JS<->torch cross-check fixture."""
    n = hand["n"]
    win = max(1, int(round(hand["input_frac"] * n)))
    off = (n - win) // 2
    idx = np.linspace(0, len(labels) - 1, N_FIXTURE, dtype=int)

    cases = [{
        "index": int(i),
        "label": int(labels[i]),
        "image28": np.round(digits28[i] * 255.0).astype(np.uint8).ravel().tolist(),
        "logits": [round(float(v), 9) for v in logits[i]],
        "pred": int(preds[i]),
    } for i in idx]

    # A few full canvases pin the JS bilinear resize to torch's align_corners=False
    # convention -- the one place a silent off-by-half-a-pixel would poison every
    # prediction downstream.
    resize_cases = [{
        "index": int(i),
        "image28": np.round(digits28[i] * 255.0).astype(np.uint8).ravel().tolist(),
        "window": [round(float(v), 8) for v in canvases[i][off:off + win, off:off + win].ravel()],
    } for i in idx[:N_RESIZE_CASES]]

    fixture = {
        "description": (
            "Reference logits from the trained torch D2NN (exports/d2nn_phase2.pt) for "
            "frozen MNIST test digits, plus reference input canvases. Generated by "
            "apps/export_d2nn_web.py; consumed by tests/test_d2nn_crosscheck.py."
        ),
        "geometry": {k: hand[k] for k in
                     ("n", "dx", "wavelength", "separation", "n_layers",
                      "readout_gain", "phase_scale", "input_frac")},
        "window": {"size": win, "offset": off},
        "cases": cases,
        "resize_cases": resize_cases,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh)
    return os.path.getsize(path)


def main():
    print(f"reading {os.path.relpath(H5_PATH, _REPO)}")
    hand = load_handoff()
    print(f"  n={hand['n']} layers={hand['n_layers']} dx={hand['dx']:g} m "
          f"lambda={hand['wavelength']:g} m z={hand['separation']:g} m")

    digits28 = recover_mnist(hand)
    model = load_model(hand)
    logits, preds = torch_logits(model, hand["canvases"], hand["phase_scale"])
    labels = hand["labels"]
    acc = float((preds == labels).mean())
    print(f"  torch accuracy on the frozen test set: {acc:.4f}")

    regions = default_regions(hand["n"], 10)
    gallery_idx = pick_gallery(labels, preds)
    n_wrong = sum(1 for i in gallery_idx if preds[i] != labels[i])
    size = write_weights_js(hand, digits28, regions, gallery_idx, labels)
    print(f"wrote {os.path.relpath(WEIGHTS_JS, _REPO)} ({size // 1024} KB, "
          f"{len(gallery_idx)} gallery digits, {n_wrong} of them misclassified)")

    size = write_fixture(hand, digits28, hand["canvases"], logits, preds, labels)
    print(f"wrote {os.path.relpath(FIXTURE_JSON, _REPO)} ({size // 1024} KB, "
          f"{N_FIXTURE} cases, {N_RESIZE_CASES} resize cases)")


if __name__ == "__main__":
    main()
