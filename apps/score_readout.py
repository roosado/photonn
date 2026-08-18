"""Score detector-box layouts against a trained D2NN's cached detector field.

Answers one question: with the phase masks held fixed, is the shipped readout
geometry (``field_frac=0.75, patch_frac=0.11``) the best one available, or is the
light that misses the detector boxes recoverable headroom?

With the masks fixed the detector-plane field is fixed too, so every layout can
be scored against **one** forward pass per image -- the layouts differ only in
which pixels each class sums over. That is what makes a 48-layout grid cheap
enough to re-run whenever the masks change.

**Selection runs on held-out data that is not the frozen test set.** Choosing a
layout is model selection, and ``photonn.train.split_dataset`` exists precisely
because model selection must not read the 2 000 frozen images every downstream
accuracy number is quoted from. The shipped model trained on the full 60 000
MNIST training images, so no part of the training split is held out; the honest
selection set is the **8 000 t10k images the frozen subset did not take**
(:func:`holdout_split`). Frozen-set accuracies are reported alongside, clearly
labelled, so the winner can be compared with the published number -- but the
ranking is the holdout's.

    python -m apps.score_readout                                   # shipped 5-mask
    python -m apps.score_readout --ckpt exports/sweep/d2nn_L56_60k_e25.pt
    python -m apps.score_readout --ckpt exports/d2nn_phase2_v1_12k.pt

Writes a JSON record per run (``--out``) with every layout's accuracy, captured
fraction, and overlap diagnostics. See ``docs/phase2_dnn.md`` (issue #5).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from photonn.detect import default_regions
from photonn.models import D2NN
from photonn.train import Dataset, encode_input, load_dataset

_REPO = Path(__file__).resolve().parent.parent

# The layout grid: 6 field fractions x 8 patch fractions = 48 layouts. The
# shipped pair (0.75, 0.11) is a grid point, so it is ranked by the same code as
# every alternative rather than being quoted from elsewhere.
FIELD_FRACS = (0.55, 0.65, 0.75, 0.85, 0.95, 1.00)
PATCH_FRACS = (0.05, 0.08, 0.11, 0.15, 0.20, 0.26, 0.32, 0.40)
SHIPPED = (0.75, 0.11)

# The frozen test set every published accuracy is quoted from, and which layout
# selection must therefore not read (photonn.train.split_dataset's docstring).
FROZEN_N = 2000
FROZEN_SEED = 0


def rebuild(ckpt):
    """Rebuild the trained D2NN from its checkpoint's recorded ``args``.

    Detector fractions are *not* recorded in the checkpoint -- the layout is
    baked into the ``readout_masks`` buffer, which ``load_state_dict``
    overwrites. Scoring therefore happens outside the model, from
    :meth:`~photonn.models.D2NN.output_field`, and never touches that buffer.
    """
    a = ckpt["args"]
    model = D2NN(n=a["grid"], n_layers=a["layers"], dx=a["dx"], wavelength=a["wavelength"],
                 separation=a["separation"], mask_init_std=a["mask_init_std"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, a


def holdout_split() -> Dataset:
    """The t10k images the frozen 2 000-image subset did *not* take.

    ``load_dataset(..., subset=2000, split="test", subset_seed=0)`` draws its
    subset with ``default_rng(0).choice``, so the complement is reproducible
    exactly. These 8 000 images are held out of training (they are test-split
    images) *and* disjoint from the frozen set, which makes them the only data in
    the project that can rank a design without spending the frozen set.
    """
    full = load_dataset("mnist", split="test")
    frozen_idx = np.random.default_rng(FROZEN_SEED).choice(
        len(full.labels), size=FROZEN_N, replace=False)
    keep = np.setdiff1d(np.arange(len(full.labels)), frozen_idx)
    return Dataset(images=full.images[keep], labels=full.labels[keep],
                   name=full.name, split=f"test-holdout[{len(keep)}]")


def overlap_stats(regions, n: int) -> dict:
    """How much of the detector plane a layout double-counts.

    ``default_regions`` places one patch per lattice cell, so patches overlap
    once the patch grows past the cell. Overlapping boxes let one pixel of light
    contribute to several classes at once, which is the mechanism the old study
    inferred from the accuracy fall-off; here it is measured directly.
    """
    cover = np.zeros((n, n), dtype=np.int32)
    for reg in regions:
        cover[reg.slices] += 1
    covered = cover > 0
    return {
        "max_cover": int(cover.max()),
        "area_frac": float(covered.mean()),
        "double_counted_frac": float((cover > 1).sum() / max(1, covered.sum())),
    }


def layout_grid(n: int, field_fracs=FIELD_FRACS, patch_fracs=PATCH_FRACS):
    """Every ``(field_frac, patch_frac)`` layout, with its regions and geometry."""
    layouts = []
    for f in field_fracs:
        for p in patch_fracs:
            regions = default_regions(n, 10, field_frac=f, patch_frac=p)
            row = {
                "field_frac": f,
                "patch_frac": p,
                "patch_px": regions[0].y1 - regions[0].y0,
                # 10 classes tile as 3 rows x 4 columns, so the *column* cell is the
                # narrower one and is what the patch has to fit inside to stay clear
                # of its neighbours. Overlap starts when patch_px exceeds it.
                "cell_px": f * n / 4.0,
                "shipped": (f, p) == SHIPPED,
            }
            row.update(overlap_stats(regions, n))
            layouts.append((row, regions))
    return layouts


@torch.no_grad()
def score_layouts(model, dataset: Dataset, layouts, *, scheme: str, batch: int = 128):
    """Accuracy and captured fraction of every layout, from one forward pass.

    All layouts' region masks are stacked into a single ``(n_layouts * 10, n*n)``
    matrix, so the per-batch readout is one matmul against the detector-plane
    intensity. Normalising by total power is monotone per sample and so cannot
    change an argmax; it is applied only where the captured fraction needs it.
    """
    n, n_layouts = model.n, len(layouts)
    stack = torch.zeros(n_layouts * 10, n, n)
    for k, (_, regions) in enumerate(layouts):
        for c, reg in enumerate(regions):
            stack[k * 10 + c][reg.slices] = 1.0
    flat = stack.reshape(n_layouts * 10, n * n)

    images = torch.as_tensor(dataset.images, dtype=torch.float32)
    labels = torch.as_tensor(dataset.labels, dtype=torch.long)
    correct = np.zeros(n_layouts, dtype=np.int64)
    captured = np.zeros(n_layouts, dtype=np.float64)

    for i in range(0, len(dataset), batch):
        field = encode_input(images[i:i + batch], scheme=scheme, n=n)
        out = model.output_field(field)
        inten = (out.real ** 2 + out.imag ** 2).reshape(field.shape[0], -1)
        sums = (inten @ flat.T).reshape(field.shape[0], n_layouts, 10)
        total = inten.sum(dim=1).clamp_min(1e-12)

        pred = sums.argmax(dim=2)                                   # (B, n_layouts)
        correct += (pred == labels[i:i + batch, None]).sum(dim=0).numpy()
        captured += (sums.sum(dim=2) / total[:, None]).sum(dim=0).numpy()

    return correct / len(dataset), captured / len(dataset)


def main():
    p = argparse.ArgumentParser(description="Re-score the detector-box layout grid.")
    p.add_argument("--ckpt", default=str(_REPO / "exports" / "d2nn_phase2.pt"))
    p.add_argument("--out", default=None, help="JSON record (default: exports/readout_<stem>.json)")
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, cfg = rebuild(ckpt)
    layouts = layout_grid(model.n)
    print(f"{Path(args.ckpt).name}: {model.n_layers} masks, {model.n}^2 grid, "
          f"scheme={cfg['scheme']}, {len(layouts)} layouts")

    holdout = holdout_split()
    frozen = load_dataset("mnist", subset=FROZEN_N, split="test", subset_seed=FROZEN_SEED)
    print(f"selection set: {len(holdout)} held-out t10k images "
          f"(disjoint from the {len(frozen)} frozen ones)")

    acc_h, cap_h = score_layouts(model, holdout, layouts, scheme=cfg["scheme"], batch=args.batch)
    acc_f, cap_f = score_layouts(model, frozen, layouts, scheme=cfg["scheme"], batch=args.batch)

    rows = []
    for k, (row, _) in enumerate(layouts):
        rows.append({**row, "acc_holdout": float(acc_h[k]), "acc_frozen": float(acc_f[k]),
                     "captured_holdout": float(cap_h[k]), "captured_frozen": float(cap_f[k])})

    order = np.argsort(-acc_h)                      # ranked by the holdout, never the frozen set
    ship = next(k for k, r in enumerate(rows) if r["shipped"])
    rank = int(np.where(order == ship)[0][0]) + 1

    print(f"\nshipped {SHIPPED}: holdout {acc_h[ship]:.4f}, frozen {acc_f[ship]:.4f}, "
          f"captured {cap_h[ship]:.3f} -- rank {rank} of {len(rows)}")
    print(f"\n{'field':>6} {'patch':>6} {'px':>4} {'cover':>5} {'holdout':>8} {'frozen':>7} {'cap':>6}")
    for k in order[:8]:
        r = rows[k]
        print(f"{r['field_frac']:>6.2f} {r['patch_frac']:>6.2f} {r['patch_px']:>4d} "
              f"{r['max_cover']:>5d} {r['acc_holdout']:>8.4f} {r['acc_frozen']:>7.4f} "
              f"{r['captured_holdout']:>6.3f}" + ("   <- shipped" if r["shipped"] else ""))

    out = Path(args.out) if args.out else _REPO / "exports" / f"readout_{Path(args.ckpt).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ckpt": str(Path(args.ckpt).name),
        "n_layers": model.n_layers, "grid": model.n, "scheme": cfg["scheme"],
        "selection_set": holdout.split, "frozen_n": len(frozen),
        "shipped": {"field_frac": SHIPPED[0], "patch_frac": SHIPPED[1], "rank": rank},
        "layouts": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
