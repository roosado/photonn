"""Export a trained D2NN for the browser demo, plus its cross-check fixture.

Bridges a trained model to ``apps/web/d2nn.js``. Reads a Phase-2 handoff and its
torch checkpoint, and writes:

- a **weight bundle** (by default ``apps/web/d2nn_weights.js``): phase masks,
  geometry, operating point, detector regions, a small gallery of frozen test
  digits, and a ``provenance`` block. **These files are committed**: ``*.h5`` and
  ``*.pt`` are gitignored, so a bundle is the only in-repo copy of a trained
  model and the only way ``apps.build_site`` can rebuild a page from a fresh
  clone.
- ``tests/fixtures/d2nn_reference.json`` -- the authoritative torch logits for a
  frozen set of digits, consumed by ``tests/test_d2nn_crosscheck.py``. Written
  only for the default 5-mask bundle; any other model is cross-checked against
  torch from its own committed bundle instead (``tests/test_deep_model.py``),
  which needs no gitignored file.

Every value comes from the handoff file; nothing about the operating point is
hardcoded here. Run from the repo root in the project venv::

    # the 5-mask model the study characterises, 8-bit, with its fixture
    python -m apps.export_d2nn_web --bits 8 --label "5 masks"

    # the 56-mask model, 4-bit, on the 5-mask model's gallery
    python -m apps.export_d2nn_web \\
        --h5 exports/sweep/d2nn_L56_60k_e25.h5 \\
        --pt exports/sweep/d2nn_L56_60k_e25.pt \\
        --out apps/web/d2nn_deep_weights.js \\
        --bits 4 --unshipped --label "56 masks" \\
        --gallery-from apps/web/d2nn_weights.js \\
        --caveat "buying those points costs 2x tighter phase control and 4.7x lower loss per mask"

``--label`` is the column title, and the convention is to name a model by its
depth ("5 masks", "56 masks") rather than by its status. ``--unshipped`` records
which model the study characterises; it is provenance metadata and a guard
against two bundles both claiming to be the headline, not something a caption
prints. What a reader is told about a non-headline model is its ``--caveat``.

At float32 a 56-mask bundle is ~4.9 MB of base64; at 8 bits it is ~1.2 MB, which
is why a deep model is exported quantised. See :mod:`apps.web_bundle` for why
that is the more faithful model rather than a shortcut.

The frozen ``/test_set`` is stored at grid resolution (the 128x128 embedded
canvas), not as 28x28 digits, so the gallery re-derives the original MNIST
digits with :func:`photonn.train.load_dataset`. That reproduction is *asserted*
against the handoff before anything is written -- a changed MNIST cache would
otherwise silently ship a gallery whose labels no longer match the frozen set.
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap

import h5py
import numpy as np
import torch

from apps.web_bundle import b64, encode_masks, js_global, quantise_phase, read_bundle
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Export a trained D2NN for the browser.")
    p.add_argument("--h5", default=H5_PATH, help="Phase-2 handoff to export (default: the shipped one)")
    p.add_argument("--pt", default=PT_PATH, help="torch checkpoint matching --h5")
    p.add_argument("--out", default=WEIGHTS_JS, help="weight bundle to write")
    p.add_argument("--bits", type=int, default=32, choices=(4, 8, 32),
                   help="phase encoding: 32 = float32 radians, 8 = uint8 codes (default: 32)")
    p.add_argument("--fixture", default=None,
                   help="path for the torch cross-check fixture. Default: the shipped "
                        "fixture when writing the shipped bundle, none otherwise.")
    p.add_argument("--gallery-from", default=None, metavar="BUNDLE",
                   help="copy the gallery from an existing bundle instead of picking one. "
                        "Required for any model meant to be compared against another: two "
                        "models shown different digits are not a comparison.")
    p.add_argument("--label", default="5 masks",
                   help="column title in the comparison widget; name a model by its depth "
                        "rather than its status (default: the default model's own depth)")
    p.add_argument("--unshipped", action="store_true",
                   help="record that this is not the model the study characterises. Provenance "
                        "only: no caption prints it. What a reader is told is --caveat.")
    p.add_argument("--caveat", default=None,
                   help="what this model's number costs, rendered into its caption; "
                        "required with --unshipped")
    args = p.parse_args(argv)

    if args.unshipped and not args.caveat:
        p.error("--unshipped needs --caveat: a model a visitor can operate invites its "
                "number being quoted, so it has to state what it is not.")
    if args.fixture is None and os.path.abspath(args.out) == os.path.abspath(WEIGHTS_JS):
        args.fixture = FIXTURE_JSON
    return args


def _rel(path: str) -> str:
    """Repo-relative, forward-slashed -- these go into generated file headers."""
    return os.path.relpath(os.path.abspath(path), _REPO).replace(os.sep, "/")


def load_handoff(h5_path=H5_PATH):
    """Read geometry, operating point, masks and the frozen test set from the h5."""
    with h5py.File(h5_path, "r") as f:
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


def load_model(hand, pt_path=PT_PATH):
    """Rebuild the trained D2NN from the handoff geometry + the torch checkpoint.

    Returns the model and the checkpoint's ``history``, which is where the
    training protocol in the bundle's ``provenance`` comes from -- the numbers
    the run actually used, not a description of it written by hand.
    """
    model = D2NN(
        n=hand["n"], n_layers=hand["n_layers"], dx=hand["dx"],
        wavelength=hand["wavelength"], separation=hand["separation"],
        readout_gain=hand["readout_gain"],
    )
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # The checkpoint must be the same model the handoff describes.
    got = np.stack([m.phi.detach().numpy() for m in model.masks]).astype("f8")
    if not np.array_equal(got, hand["masks"]):
        raise RuntimeError("checkpoint phase masks differ from the handoff's phase_masks.")
    return model, ckpt.get("history", {})


def provenance(history, accuracy, n_test, label="Shipped", shipped=True, caveat=None):
    """Describe where this model's number came from, for the widgets to print.

    The comparison board renders its captions from this block rather than from
    literals in JavaScript, so promoting a model is regenerating a bundle. Every
    field is measured here: the accuracy is the run just scored above, and the
    protocol is read off the checkpoint. A missing field is an error rather than
    a default -- a plausible-looking wrong protocol is worse than a crash.

    ``scored_on`` is written the same way for every model this exporter touches,
    which is what lets the widget tell "measured differently" from "measured
    identically" without being told which is which.
    """
    missing = [k for k in ("n_samples", "epochs", "seed") if k not in history]
    if missing:
        raise RuntimeError(
            f"checkpoint history lacks {missing}; refusing to write a provenance block "
            "with invented training parameters."
        )
    prov = {
        "label": label,
        "accuracy": float(accuracy),
        "scored_on": f"the frozen {n_test:,}-image MNIST test set",
        "shipped": shipped,
        "protocol": {
            "n_train": int(history["n_samples"]),
            "epochs": int(history["epochs"]),
            "seed": int(history["seed"]),
        },
    }
    if caveat:
        prov["caveat"] = caveat
    return prov


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


def gallery_payload(digits28, labels, preds, source=None):
    """The gallery fields of a bundle, either picked here or copied from another.

    A second model must be shown the *same* digits as the one it is compared
    against, or the board is two demos side by side rather than a comparison --
    and the default gallery is deliberately stocked with digits the 5-mask model
    gets wrong, which is exactly what makes the contrast visible. Copying is
    therefore the right default for any second model, and it is a flag rather
    than a guess.
    """
    if source is not None:
        other = read_bundle(source)
        return {k: other[k] for k in ("gallery_b64", "gallery_labels", "gallery_size")}, None

    idx = pick_gallery(labels, preds)
    return {
        "gallery_b64": b64(np.round(digits28[idx] * 255.0).astype(np.uint8), "u1"),
        "gallery_labels": [int(labels[i]) for i in idx],
        "gallery_size": 28,
    }, idx


def bundle_header(out_path, h5_path, hand, bits, n_gallery, prov, max_err, copied_from):
    """The comment block at the top of a generated bundle."""
    name = os.path.basename(out_path)
    what = ("the trained Phase-2 diffractive network" if prov["shipped"]
            else f"the {hand['n_layers']}-mask diffractive network")

    if bits in (4, 8):
        levels = 1 << bits
        packing = ("uint8 phase codes" if bits == 8
                   else "4-bit phase codes packed two per byte, high nibble first")
        masks_line = (
            f" * masks_b64   : {hand['n_layers']} x {hand['n']} x {hand['n']} {packing} "
            f"(masks_bits: {bits}), decoded as\n"
            f" *               code * 2pi/{levels} - pi radians and applied as E * exp(i * phi).\n"
            f" *               Max encoding error {max_err:.4f} rad; the Phase-4 budget holds this\n"
            f" *               design to 3-bit phase control, so {bits} bits is within what the\n"
            f" *               hardware model already assumes.\n")
    else:
        masks_line = (
            f" * masks_b64   : {hand['n_layers']} x {hand['n']} x {hand['n']} float32 phase "
            f"(radians, wrapped to [-pi, pi)),\n"
            f" *               little-endian, applied as E * exp(i * phi).\n")

    gallery_line = f" * gallery_b64 : {n_gallery} x 28 x 28 uint8 MNIST digits from the frozen test set.\n"
    if copied_from:
        gallery_line += (f" *               Copied from {_rel(copied_from)} so both models are shown\n"
                         f" *               the same digits and the comparison is a comparison.\n")

    # What the caption will say, restated for whoever opens the file: the number
    # and what it cost. Which model the study characterises is provenance
    # (`shipped`), deliberately not part of how either one is described.
    note = ""
    if prov.get("caveat"):
        said = (f"{prov['accuracy']:.4f} on {prov['scored_on']} "
                f"({prov['protocol']['n_train']:,} images, {prov['protocol']['epochs']} epochs). "
                f"{prov['caveat'][0].upper()}{prov['caveat'][1:]}.")
        note = " *\n" + "".join(f" * {line}\n" for line in textwrap.wrap(said, 74))

    return (
        f"/*\n"
        f" * {name} -- {what}, for the browser.\n"
        f" *\n"
        f" * GENERATED by apps/export_d2nn_web.py from {_rel(h5_path)} -- do not edit\n"
        f" * by hand; re-run the exporter instead. This file is committed because the h5 and\n"
        f" * pt exports are gitignored, so it is the repo's only copy of the trained model.\n"
        f"{note}"
        f" *\n"
        f"{masks_line}"
        f"{gallery_line}"
        f" * regions     : ten [y0, y1, x0, x1] detector boxes, class order, from\n"
        f" *               photonn.detect.default_regions.\n"
        f" * provenance  : where this model's number came from. The comparison widget\n"
        f" *               renders its captions from this, so no accuracy is written into\n"
        f" *               JavaScript and changing what a model claims is regenerating a\n"
        f" *               bundle. `shipped` records which model the study characterises;\n"
        f" *               no caption reads it, and a column is named for its depth.\n"
        f" * All lengths in metres.\n"
        f" */\n"
    )


def write_weights_js(hand, regions, gallery, prov, masks_b64, bits, path, header):
    """Write the browser weight bundle."""
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
    }
    if bits != 32:
        payload["masks_bits"] = bits
    payload["masks_b64"] = masks_b64
    payload.update(gallery)
    payload["provenance"] = prov

    body = ",\n    ".join(f'"{k}": {json.dumps(v)}' for k, v in payload.items())
    js = f"""{header}(function () {{
  "use strict";
  var W = {{
    {body}
  }};
  if (typeof module !== "undefined" && module.exports) module.exports = W;
  if (typeof window !== "undefined") window.{js_global(path)} = W;
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


def main(argv=None):
    args = parse_args(argv)

    print(f"reading {_rel(args.h5)}")
    hand = load_handoff(args.h5)
    print(f"  n={hand['n']} layers={hand['n_layers']} dx={hand['dx']:g} m "
          f"lambda={hand['wavelength']:g} m z={hand['separation']:g} m")

    digits28 = recover_mnist(hand)
    model, history = load_model(hand, args.pt)

    # Score the model that is being *shipped*, not the float model it came from.
    # At --bits 8 or 4 the browser applies quantised phases, so the float logits
    # would describe a network no visitor ever runs -- and the cross-check fixture
    # written below would pin the wrong one.
    if args.bits != 32:
        quantised = quantise_phase(hand["masks"], args.bits)
        with torch.no_grad():
            for m, phi in zip(model.masks, quantised):
                m.phi.copy_(torch.from_numpy(phi).to(m.phi.dtype))

    logits, preds = torch_logits(model, hand["canvases"], hand["phase_scale"])
    labels = hand["labels"]
    acc = float((preds == labels).mean())
    print(f"  torch accuracy on the frozen test set: {acc:.4f}"
          + ("" if args.bits == 32 else f"  ({args.bits}-bit phase)"))

    prov = provenance(history, acc, len(labels), label=args.label,
                      shipped=not args.unshipped, caveat=args.caveat)
    print(f"  provenance: {prov['accuracy']} on {prov['scored_on']}, "
          f"{prov['protocol']['n_train']} images x {prov['protocol']['epochs']} epochs")

    regions = default_regions(hand["n"], 10)
    gallery, gallery_idx = gallery_payload(digits28, labels, preds, args.gallery_from)
    n_gallery = len(gallery["gallery_labels"])
    masks_b64, max_err = encode_masks(hand["masks"], args.bits)
    header = bundle_header(args.out, args.h5, hand, args.bits, n_gallery, prov,
                           max_err=max_err, copied_from=args.gallery_from)
    size = write_weights_js(hand, regions, gallery, prov, masks_b64, args.bits, args.out, header)

    detail = f"{n_gallery} gallery digits"
    if gallery_idx is not None:
        detail += f", {sum(1 for i in gallery_idx if preds[i] != labels[i])} of them misclassified"
    else:
        detail += f" copied from {_rel(args.gallery_from)}"
    if args.bits != 32:
        detail += f", {args.bits}-bit phase (max error {max_err:.4f} rad)"
    print(f"wrote {_rel(args.out)} ({size // 1024} KB, {detail})")

    if args.fixture:
        size = write_fixture(hand, digits28, hand["canvases"], logits, preds, labels, args.fixture)
        print(f"wrote {_rel(args.fixture)} ({size // 1024} KB, "
              f"{N_FIXTURE} cases, {N_RESIZE_CASES} resize cases)")


if __name__ == "__main__":
    main()
