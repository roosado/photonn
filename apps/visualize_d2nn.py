"""Render a trained D2NN: its phase masks and an example input -> output.

Loads the checkpoint written by ``apps.train_d2nn`` (default
``exports/d2nn_phase2.pt``), rebuilds the model, and saves a figure showing each
trained phase mask (wrapped to [-pi, pi]) plus a sample input intensity and the
detector-plane output intensity with the detector regions overlaid. Supports the
"what the masks do optically" section of ``docs/phase2_dnn.md``.

    python -m apps.visualize_d2nn
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from photonn.models import D2NN
from photonn.train import encode_input, load_dataset

_REPO = Path(__file__).resolve().parent.parent


def _rebuild(ckpt):
    a = ckpt["args"]
    model = D2NN(n=a["grid"], n_layers=a["layers"], dx=a["dx"], wavelength=a["wavelength"],
                 separation=a["separation"], mask_init_std=a["mask_init_std"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, a


def main():
    p = argparse.ArgumentParser(description="Visualise a trained D2NN.")
    p.add_argument("--ckpt", default=str(_REPO / "exports" / "d2nn_phase2.pt"))
    p.add_argument("--out", default=str(_REPO / "docs" / "figures" / "phase2_masks.png"))
    p.add_argument("--digit", type=int, default=7, help="which class to show as the example")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, cfg = _rebuild(ckpt)
    scheme = cfg["scheme"]

    test = load_dataset("mnist", subset=2000, split="test")
    idx = int(np.where(test.labels == args.digit)[0][0])
    img = torch.as_tensor(test.images[idx], dtype=torch.float32)[None]
    field = encode_input(img, scheme=scheme, n=model.n)

    with torch.no_grad():
        out = model.output_field(field)[0].numpy()
        logits = model(field)[0].numpy()
    pred = int(logits.argmax())

    in_intensity = (field[0].abs() ** 2).numpy()
    out_intensity = np.abs(out) ** 2

    n_masks = model.n_layers
    fig = plt.figure(figsize=(3.0 * max(n_masks, 3), 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, n_masks)

    # Top row: trained phase masks, wrapped to [-pi, pi].
    for k, mask in enumerate(model.masks):
        ax = fig.add_subplot(gs[0, k])
        phi = np.angle(np.exp(1j * mask.phi.detach().numpy()))
        im = ax.imshow(phi, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"mask {k + 1}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=fig.axes[:n_masks], shrink=0.7, label="phase (rad)")

    # Bottom row: input intensity, output intensity (+detectors), logits.
    ax_in = fig.add_subplot(gs[1, 0])
    ax_in.imshow(in_intensity, cmap="inferno")
    ax_in.set_title(f"input |E|²  (digit {args.digit})", fontsize=10)
    ax_in.set_xticks([]); ax_in.set_yticks([])

    ax_out = fig.add_subplot(gs[1, 1])
    ax_out.imshow(out_intensity, cmap="inferno")
    ax_out.set_title(f"detector plane |E|²  (pred {pred})", fontsize=10)
    ax_out.set_xticks([]); ax_out.set_yticks([])
    for reg in model.regions:
        correct = reg.label == args.digit
        ax_out.add_patch(Rectangle(
            (reg.x0, reg.y0), reg.x1 - reg.x0, reg.y1 - reg.y0,
            fill=False, edgecolor="lime" if correct else "white",
            linewidth=1.5 if correct else 0.6))

    ax_bar = fig.add_subplot(gs[1, 2]) if n_masks >= 3 else fig.add_subplot(gs[1, min(2, n_masks - 1)])
    ax_bar.bar(range(model.n_classes), logits, color="#4c78a8")
    ax_bar.axvline(args.digit, color="lime", lw=1.5, label="true")
    ax_bar.set_title("detector logits", fontsize=10)
    ax_bar.set_xlabel("class"); ax_bar.set_xticks(range(model.n_classes))
    ax_bar.legend(fontsize=8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}  (example digit {args.digit} -> predicted {pred})")


if __name__ == "__main__":
    main()
