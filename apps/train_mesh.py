"""Phase-3 deliverable: train an MZI-mesh classifier and export it to the handoff.

Trains a 36-mode SVD mesh (U.Sigma.V*, two universal Clements meshes around a
diagonal) to classify MNIST downsampled to 6x6, reports the direct comparison to
the Phase-2 D2NN (parameter count, depth, footprint), and serialises the trained
MZI phases through the one-directional HDF5 handoff (mesh path).

Run (from the repo root, in the project venv)::

    python -m apps.train_mesh                 # deliverable config
    python -m apps.train_mesh --quick         # fast smoke config

Deterministic given ``--seed`` (recorded in the export).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from photonn.export import validate_handoff, write_handoff
from photonn.models import MeshNetwork
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
    p.add_argument("--out-h5", default=str(_REPO / "exports" / "mesh_phase3.h5"))
    p.add_argument("--out-pt", default=str(_REPO / "exports" / "mesh_phase3.pt"))
    return p.parse_args()


def main():
    args = parse_args()
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

    # Store the MZI phases: theta/phi concatenated as [V mesh, U mesh]. Sigma and
    # output phases are additional (a full as-built reconstruction of the SVD mesh
    # would extend the schema -- the mesh error budget is a documented follow-on).
    theta = np.concatenate([model.v.theta.detach().numpy(), model.u.theta.detach().numpy()])
    phi = np.concatenate([model.v.phi.detach().numpy(), model.u.phi.detach().numpy()])
    test_maps = encode_modes(torch.as_tensor(test_ds.images, dtype=torch.float32), n_modes=n)
    test_images = test_maps.abs().numpy().astype("f4").reshape(-1, int(n ** 0.5), int(n ** 0.5))

    write_handoff(
        out_h5,
        model_type="mesh",
        parameters={"phase_theta": theta.astype("f8"), "phase_phi": phi.astype("f8")},
        geometry={
            "grid_size": int(round(n ** 0.5)),
            "physical_extent_m": 0.0,
            "n_layers": 2,                     # V and U meshes
            "layer_separations_m": np.zeros(1, dtype="f8"),
        },
        operating_point={"wavelength_m": args.wavelength, "n_modes": n,
                         "n_classes": args.classes, "readout_gain": model.readout_gain},
        test_images=test_images,
        test_labels=test_ds.labels,
        description=(
            f"MZI mesh Phase 3 | modes={n} | SVD U*Sigma*V* | test_acc={test_acc:.4f} | "
            f"seed={args.seed} | phase_theta/phi = concat[V, U] MZI phases "
            f"({n * (n - 1) // 2} each); Sigma + output phases not in schema 0.1.0"
        ),
    )
    validate_handoff(out_h5)
    torch.save({"state_dict": model.state_dict(), "history": hist, "args": vars(args)}, out_pt)
    print(f"\nexported handoff -> {out_h5}  (validated)")
    print(f"saved torch model -> {out_pt}")


if __name__ == "__main__":
    main()
