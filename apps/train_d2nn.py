"""Phase-2 deliverable: train a diffractive D2NN and export it to the handoff.

Trains a stack of phase masks to classify MNIST (input encoded in **both**
amplitude and phase), reports the optical power budget (photons per detector
region per inference), then serialises the trained model to the one-directional
HDF5 handoff the MATLAB as-built model reads.

Run (from the repo root, in the project venv)::

    python -m apps.train_d2nn                 # deliverable config
    python -m apps.train_d2nn --quick         # fast smoke config
    python -m apps.train_d2nn --epochs 30 --subset-train 60000   # full run

Everything is deterministic given ``--seed`` (recorded in the export).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from photonn import detect
from photonn.export import validate_handoff, write_handoff
from photonn.fields import Field
from photonn.models import D2NN
from photonn.train import embed_input, encode_input, evaluate, load_dataset, train

_REPO = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="Train and export a Phase-2 D2NN.")
    p.add_argument("--grid", type=int, default=128, help="field grid size N (N x N)")
    p.add_argument("--layers", type=int, default=5, help="number of trainable phase masks")
    p.add_argument("--dx", type=float, default=8e-6, help="pixel pitch (m); typical SLM ~8 um")
    p.add_argument("--wavelength", type=float, default=532e-9, help="operating wavelength (m)")
    p.add_argument("--separation", type=float, default=3e-3, help="inter-plane distance (m)")
    p.add_argument("--scheme", default="both", choices=["amplitude", "phase", "both"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--subset-train", type=int, default=12000, help="train subset (None-like: use 60000)")
    p.add_argument("--subset-test", type=int, default=2000, help="test subset (frozen, exported)")
    p.add_argument("--mask-init-std", type=float, default=0.3, help="phase-mask init std (rad)")
    p.add_argument("--seed", type=int, default=20260724)
    # Illustrative operating point for the photon budget (design choices, not error magnitudes):
    p.add_argument("--input-power-w", type=float, default=1e-3, help="optical power at the input (W)")
    p.add_argument("--integration-time-s", type=float, default=1e-3, help="detector integration time (s)")
    p.add_argument("--quick", action="store_true", help="tiny/fast config for a smoke run")
    p.add_argument("--out-h5", default=str(_REPO / "exports" / "d2nn_phase2.h5"))
    p.add_argument("--out-pt", default=str(_REPO / "exports" / "d2nn_phase2.pt"))
    return p.parse_args()


def report_power_budget(model, field, args):
    """Photons per detector region per inference, averaged over a batch."""
    dx, lam = args.dx, args.wavelength
    with torch.no_grad():
        out = model.output_field(field).cpu().numpy()  # (B, n, n) complex

    # Each encoded input is normalised to sum|E|^2 = 1, so its optical power is
    # exactly dx^2; use that as the photon-budget reference (see photon_budget).
    ref_power = dx * dx
    per_region = []
    captured = []
    for k in range(out.shape[0]):
        pb = detect.photon_budget(
            Field(out[k], dx, lam),
            input_power_w=args.input_power_w,
            integration_time_s=args.integration_time_s,
            regions=model.regions,
            reference_power=ref_power,
        )
        per_region.append(pb.per_region)
        captured.append(pb.captured_fraction)
        n_in, e_ph = pb.photons_in, pb.photon_energy_j

    per_region = np.array(per_region)  # (B, C)
    print("\n--- optical power budget (per inference) ---")
    print(f"  operating point : {args.input_power_w * 1e3:g} mW x {args.integration_time_s * 1e3:g} ms"
          f"  at {lam * 1e9:g} nm")
    print(f"  photon energy   : {e_ph:.3e} J")
    print(f"  photons in      : {n_in:.3e}")
    print(f"  captured in detectors: {np.mean(captured) * 100:.1f}% of input photons")
    print(f"  photons / detector region : mean {per_region.mean():.3e}, "
          f"min {per_region.min():.3e}, max {per_region.max():.3e}")
    winner = per_region.max(axis=1)  # brightest (predicted) detector
    print(f"  photons in the winning detector : mean {winner.mean():.3e}")
    return per_region


def main():
    args = parse_args()
    if args.quick:
        args.grid, args.layers, args.epochs = 64, 4, 4
        args.subset_train, args.subset_test = 2000, 1000

    torch.manual_seed(args.seed)  # reproducible phase-mask init
    print(f"Phase-2 D2NN | grid={args.grid} layers={args.layers} scheme={args.scheme} "
          f"seed={args.seed}")

    train_ds = load_dataset("mnist", subset=args.subset_train, split="train")
    test_ds = load_dataset("mnist", subset=args.subset_test, split="test")
    print(f"train={len(train_ds)}  test={len(test_ds)}")

    model = D2NN(
        n=args.grid, n_layers=args.layers, dx=args.dx, wavelength=args.wavelength,
        separation=args.separation, mask_init_std=args.mask_init_std,
    )
    print(f"trainable params: {sum(p.numel() for p in model.parameters())}")

    model, hist = train(
        model, train_ds, epochs=args.epochs, seed=args.seed, lr=args.lr,
        batch_size=args.batch, scheme=args.scheme, val_dataset=test_ds,
    )
    test_acc = evaluate(model, test_ds, scheme=args.scheme)
    print(f"\nfinal test accuracy: {test_acc:.4f}  (chance = {1.0 / model.n_classes:.3f})")

    # Power budget on a representative test batch.
    imgs = torch.as_tensor(test_ds.images[:args.batch], dtype=torch.float32)
    field = encode_input(imgs, scheme=args.scheme, n=args.grid)
    report_power_budget(model, field, args)

    # -- export --------------------------------------------------------------
    out_h5 = Path(args.out_h5)
    out_pt = Path(args.out_pt)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    phase_masks = np.stack([m.phi.detach().cpu().numpy() for m in model.masks]).astype("f8")

    # Frozen test set at grid resolution: the RAW embedded input map (canvas in
    # [0,1], before complex encoding/normalisation). The as-built (MATLAB) model
    # recovers the field as E = img * exp(i * phase_scale * img) for
    # scheme="both"; that phase uses the raw canvas values, so the raw map (not
    # the unit-normalised one) is what must be stored. The forward pass is
    # scale-invariant, so amplitude normalisation is irrelevant to the result.
    test_maps = embed_input(
        torch.as_tensor(test_ds.images, dtype=torch.float32), n=args.grid,
    ).numpy().astype("f4")

    n_gaps = args.layers + 1  # input->m1, inter-mask, mL->detector; all uniform
    write_handoff(
        out_h5,
        model_type="d2nn",
        parameters={"phase_masks": phase_masks},
        geometry={
            "grid_size": args.grid,
            "physical_extent_m": args.grid * args.dx,
            "n_layers": args.layers,
            "layer_separations_m": np.full(n_gaps, args.separation, dtype="f8"),
        },
        operating_point={
            "wavelength_m": args.wavelength,
            "pixel_pitch_m": args.dx,
            "readout_gain": model.readout_gain,
            "phase_scale_rad": np.pi,
            "input_frac": 0.5,
            "encoding_code": {"amplitude": 0, "phase": 1, "both": 2}[args.scheme],
            "input_power_w": args.input_power_w,
            "integration_time_s": args.integration_time_s,
        },
        test_images=test_maps,
        test_labels=test_ds.labels,
        description=(
            f"D2NN Phase 2 | scheme={args.scheme} | layers={args.layers} | "
            f"test_acc={test_acc:.4f} | seed={args.seed} | "
            f"forward: propagate->phasemask x{args.layers} ->propagate->detector | "
            f"encoding_code 0=amplitude,1=phase,2=both; layer_separations has n_layers+1 uniform gaps"
        ),
    )
    validate_handoff(out_h5)
    torch.save({"state_dict": model.state_dict(), "history": hist, "args": vars(args)}, out_pt)

    print(f"\nexported handoff -> {out_h5}  (validated)")
    print(f"saved torch model -> {out_pt}")


if __name__ == "__main__":
    main()
