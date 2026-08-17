"""Phase-3 deliverable: train an MZI-mesh classifier and export it to the handoff.

Trains a 36-mode SVD mesh (U.Sigma.V*, two universal Clements meshes around a
diagonal) to classify MNIST downsampled to 6x6, reports the direct comparison to
the Phase-2 D2NN (parameter count, depth, footprint), and serialises the trained
MZI phases through the one-directional HDF5 handoff (mesh path).

Run (from the repo root, in the project venv)::

    python -m apps.train_mesh                 # deliverable config
    python -m apps.train_mesh --quick         # fast smoke config
    python -m apps.train_mesh --export-only   # re-export the handoff, no training

Deterministic given ``--seed`` (recorded in the export).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from photonn.export import validate_handoff, write_handoff
from photonn.models import MeshNetwork
from photonn.mzi import passivize
from photonn.train import encode_modes, evaluate, load_dataset, train

_REPO = Path(__file__).resolve().parent.parent

# D2NN reference (docs/phase2_dnn.md) for the comparison.
_D2NN = {"params": 5 * 128 * 128, "acc": 0.799, "depth": "5 masks + 6 propagations",
         "input": "28x28 in a 128x128 field", "footprint": "5 x 128^2 phase pixels"}


def parse_args():
    p = argparse.ArgumentParser(description="Train and export a Phase-3 MZI mesh.")
    p.add_argument("--modes", type=int, default=36, help="mesh modes (perfect square)")
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--subset-train", type=int, default=20000)
    p.add_argument("--subset-test", type=int, default=2000)
    p.add_argument("--wavelength", type=float, default=1.55e-6, help="design wavelength (m)")
    p.add_argument("--seed", type=int, default=20260725)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--export-only", action="store_true",
                   help="rebuild the handoff from --out-pt without retraining")
    p.add_argument("--out-h5", default=str(_REPO / "exports" / "mesh_phase3.h5"))
    p.add_argument("--out-pt", default=str(_REPO / "exports" / "mesh_phase3.pt"))
    return p.parse_args()


def export_handoff(model, test_ds, *, out_h5, n, n_classes, wavelength, seed, test_acc):
    """Write the mesh handoff: MZI angles, a passivized Sigma, and the output phases.

    Schema 0.2.0. Everything needed to rebuild the operator crosses the boundary --
    0.1.0 carried the MZI angles alone and so could not reproduce the ideal accuracy
    on the MATLAB side, which is the anchor the whole error budget rests on.
    """
    n_mzi = n * (n - 1) // 2
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    # Sigma is trained unconstrained and comes out signed and > 1; passivize() folds
    # the sign into V's output phase and the scale into an external gain, both of
    # which leave the logits identical. See photonn/mzi.py and docs/phase3_mesh.md.
    sigma_p, out_phase_v, sigma_gain = passivize(sd["sigma"], sd["v.out_phase"])

    # theta/phi concatenated as [V, U]; out_phase indexed the same way (export.MESH_ORDER).
    theta = np.concatenate([sd["v.theta"], sd["u.theta"]])
    phi = np.concatenate([sd["v.phi"], sd["u.phi"]])
    out_phase = np.stack([out_phase_v, sd["u.out_phase"]])

    test_maps = encode_modes(torch.as_tensor(test_ds.images, dtype=torch.float32), n_modes=n)
    g = int(round(n ** 0.5))
    test_images = test_maps.abs().numpy().astype("f4").reshape(-1, g, g)

    write_handoff(
        out_h5,
        model_type="mesh",
        parameters={"phase_theta": theta, "phase_phi": phi,
                    "sigma": sigma_p, "out_phase": out_phase},
        geometry={
            "grid_size": g,
            "physical_extent_m": 0.0,
            "n_layers": 2,                     # V and U meshes
            "layer_separations_m": np.zeros(1, dtype="f8"),
        },
        # input_power_w / integration_time_s are the photon budget the detector sweep
        # needs; state them here rather than leaning on the MATLAB reader's defaults.
        operating_point={"wavelength_m": wavelength, "n_modes": n,
                         "n_classes": n_classes, "readout_gain": model.readout_gain,
                         "sigma_gain": sigma_gain,
                         "input_power_w": 1e-3, "integration_time_s": 1e-3},
        test_images=test_images,
        test_labels=test_ds.labels,
        description=(
            f"MZI mesh Phase 3 | modes={n} | SVD U*diag(sigma)*V | test_acc={test_acc:.4f} | "
            f"seed={seed} | phase_theta/phi = concat[V, U] MZI phases ({n_mzi} each); "
            f"out_phase = [V; U]; sigma passivized to <=1 with external gain "
            f"{sigma_gain:.4f} (logit-preserving)"
        ),
    )
    validate_handoff(out_h5)
    return sigma_gain


def export_only(args):
    """Rebuild the handoff from the saved checkpoint, without retraining.

    Retraining risks moving the accuracy the error budget is anchored to, so the
    re-export reads ``--out-pt`` and leaves the trained parameters exactly as they are.
    """
    out_pt = Path(args.out_pt)
    if not out_pt.exists():
        raise SystemExit(f"checkpoint not found: {out_pt} (train first, or pass --out-pt)")
    ckpt = torch.load(out_pt, weights_only=False)
    saved = ckpt["args"]
    n, n_classes = saved["modes"], saved["classes"]
    print(f"re-exporting from {out_pt} | modes={n} classes={n_classes} seed={saved['seed']}")

    model = MeshNetwork(n, n_classes, use_svd=True)
    model.load_state_dict(ckpt["state_dict"])
    test_ds = load_dataset("mnist", subset=saved["subset_test"], split="test")
    encoder = lambda imgs: encode_modes(imgs, n_modes=n)
    test_acc = evaluate(model, test_ds, encoder=encoder)
    print(f"test accuracy: {test_acc:.4f}")

    gain = export_handoff(model, test_ds, out_h5=Path(args.out_h5), n=n, n_classes=n_classes,
                          wavelength=saved["wavelength"], seed=saved["seed"], test_acc=test_acc)
    print(f"sigma passivized (external gain {gain:.4f}, logits unchanged)")
    print(f"exported handoff -> {args.out_h5}  (validated)")


def main():
    args = parse_args()
    if args.export_only:
        return export_only(args)
    if args.quick:
        args.epochs, args.subset_train, args.subset_test = 5, 4000, 1000

    torch.manual_seed(args.seed)  # reproducible mesh init
    n = args.modes
    print(f"Phase-3 MZI mesh | modes={n} classes={args.classes} seed={args.seed}")

    train_ds = load_dataset("mnist", subset=args.subset_train, split="train")
    test_ds = load_dataset("mnist", subset=args.subset_test, split="test")
    print(f"train={len(train_ds)}  test={len(test_ds)}  (MNIST downsampled to "
          f"{int(round(n ** 0.5))}x{int(round(n ** 0.5))})")

    model = MeshNetwork(n, args.classes, use_svd=True)
    n_params = sum(p.numel() for p in model.parameters())
    n_mzi = 2 * n * (n - 1) // 2
    print(f"trainable params: {n_params}  ({n_mzi} MZIs + Sigma + output phases)")

    encoder = lambda imgs: encode_modes(imgs, n_modes=n)
    model, hist = train(model, train_ds, epochs=args.epochs, seed=args.seed, lr=args.lr,
                        batch_size=args.batch, val_dataset=test_ds, encoder=encoder)
    test_acc = evaluate(model, test_ds, encoder=encoder)
    print(f"\nfinal test accuracy: {test_acc:.4f}  (chance = {1.0 / args.classes:.3f})")

    # -- D2NN comparison -----------------------------------------------------
    print("\n--- mesh vs D2NN ---")
    print(f"  {'':16s}{'MZI mesh':>22s}{'D2NN':>26s}")
    print(f"  {'params':16s}{n_params:>22d}{_D2NN['params']:>26d}")
    print(f"  {'MNIST accuracy':16s}{test_acc:>22.3f}{_D2NN['acc']:>26.3f}")
    print(f"  {'depth':16s}{f'{2 * n} MZI layers':>22s}{_D2NN['depth']:>26s}")
    print(f"  {'footprint':16s}{f'{n_mzi} MZIs':>22s}{_D2NN['footprint']:>26s}")
    print(f"  {'input':16s}{f'{n} modes (6x6)':>22s}{_D2NN['input']:>26s}")

    # -- export (mesh handoff path) ------------------------------------------
    out_h5, out_pt = Path(args.out_h5), Path(args.out_pt)
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    gain = export_handoff(model, test_ds, out_h5=out_h5, n=n, n_classes=args.classes,
                          wavelength=args.wavelength, seed=args.seed, test_acc=test_acc)
    torch.save({"state_dict": model.state_dict(), "history": hist, "args": vars(args)}, out_pt)
    print(f"\nsigma passivized (external gain {gain:.4f}, logits unchanged)")
    print(f"exported handoff -> {out_h5}  (validated)")
    print(f"saved torch model -> {out_pt}")


if __name__ == "__main__":
    main()
