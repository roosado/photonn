"""Phase-5 deliverable: train the activated two-layer mesh and export it to the handoff.

Two SVD mesh layers on 16 Fourier modes with Williamson et al.'s (2020) electro-optic
activation between them (``docs/phase5_activation.md``), plus -- with ``--one-layer`` --
the one-layer 16-mode mesh trained under the same protocol, which is the baseline
every "change against one layer" in the Phase-5 budget is measured from. The
published 36-mode mesh is **not** that baseline: different input, different width.

Run (from the repo root, in the project venv)::

    python -m apps.train_deep_mesh                # the activated two-layer mesh
    python -m apps.train_deep_mesh --one-layer    # its one-layer baseline
    python -m apps.train_deep_mesh --export-only  # re-export from the checkpoint

**The defaults write ``exports/deep_mesh_phase5.*`` and ``exports/mesh16_phase5.*``.**
Never point them at ``exports/mesh_phase3.*``: that is the published 0.7355 model.

Deterministic given ``--seed`` (recorded in the export). The protocol is the one the
Phase-5 gate passed under (``apps/eo_gate.py``, arm ``fourier-2svd-act-passive``):
all 60 000 training images, batch 500, 20 epochs, Adam at 2e-2, scored on all 10 000
test images, which become the frozen test set.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from photonn import mzi
from photonn.export import validate_handoff, write_handoff
from photonn.models import DeepMeshNetwork, MeshNetwork
from photonn.train import encode_fourier, evaluate, load_dataset, train

_REPO = Path(__file__).resolve().parent.parent

# -- the device -------------------------------------------------------------------
# Williamson, Hughes, Minkov, Bartlett, Pai & Fan, IEEE JSTQE 26(1):7700412 (2020).
# Their values are *design* values from a simulation paper, not measurements; the
# ledger in docs/parameter_sources.md says which is which.
EO_ALPHA = 0.1                    # tapped fraction -- Williamson 2020 Sec. VI-B / Fig. 6 (design)
EO_PHI_B = math.pi                # bias phase -- Williamson 2020 Fig. 6, phi_b = 1.00 pi (design)
EO_G_NORMALISED = 0.05 * math.pi  # rad per unit of total input power -- Williamson 2020 Fig. 6
                                  # (design; "selected heuristically", their words)
INPUT_POWER_W = 1e-3              # the mesh budget's nominal input (docs/tolerance_mesh.md);
                                  # a design choice, not a measured source power
EO_G_PHI = EO_G_NORMALISED / INPUT_POWER_W   # rad/W: the paper's setting, in watts at 1 mW

# The raw device values that give that g_phi through Eq. (7). Two are cited; the
# third is whatever makes the product come out, and is flagged as such.
EO_RESPONSIVITY_A_PER_W = 1.0     # SiGe photodiode, imec iSiPP50G, ~1.0 A/W at 3 V reverse
                                  # bias: Meyer et al., Nat. Commun. 17:3396 (2026), Fig. 2e
                                  # inset -- MEASURED, read from a plot. Also Williamson 2020
                                  # Table I's design value.
EO_V_PI = 10.0                    # V -- Williamson 2020 Sec. VII, "a phase modulator V_pi of
                                  # 10 V ... experimentally feasible" (a design example; no
                                  # measured device). UNSOURCED as a measurement.
EO_V_BIAS = EO_PHI_B * EO_V_PI / math.pi     # Eq. (5) inverted: phi_b = pi at V_b = V_pi
EO_TIA_GAIN_OHM = EO_G_PHI * EO_V_PI / (math.pi * EO_ALPHA * EO_RESPONSIVITY_A_PER_W)
# = 5 kOhm (74 dB-Ohm), derived from the three above. UNSOURCED: no datasheet in this
# project shows a transimpedance amplifier of this gain at EO_BANDWIDTH_HZ.
EO_BANDWIDTH_HZ = 10e9            # Williamson 2020 Table I, "modulator and detector rate"
                                  # (design)
SYMBOL_TIME_S = 1.0 / EO_BANDWIDTH_HZ   # one input per symbol at that rate; the readout
                                        # integrates one symbol

assert math.isclose(mzi.eo_phase_gain(alpha=EO_ALPHA, tia_gain_ohm=EO_TIA_GAIN_OHM,
                                      responsivity_a_per_w=EO_RESPONSIVITY_A_PER_W,
                                      v_pi=EO_V_PI), EO_G_PHI, rel_tol=1e-12)
assert math.isclose(mzi.eo_bias_phase(v_bias=EO_V_BIAS, v_pi=EO_V_PI), EO_PHI_B,
                    rel_tol=1e-12)


def parse_args():
    p = argparse.ArgumentParser(description="Train and export the Phase-5 activated mesh.")
    p.add_argument("--modes", type=int, default=16, help="Fourier modes (mesh width)")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--subset-train", type=int, default=0, help="0 = all 60 000")
    p.add_argument("--subset-test", type=int, default=0, help="0 = all 10 000")
    p.add_argument("--wavelength", type=float, default=1.55e-6, help="design wavelength (m)")
    p.add_argument("--seed", type=int, default=20260725)
    p.add_argument("--one-layer", action="store_true",
                   help="train the one-layer 16-mode baseline instead (a 'mesh' handoff)")
    p.add_argument("--export-only", action="store_true",
                   help="rebuild the handoff from --out-pt without retraining")
    p.add_argument("--out-h5", default=None)
    p.add_argument("--out-pt", default=None)
    args = p.parse_args()
    stem = "mesh16_phase5" if args.one_layer else "deep_mesh_phase5"
    args.out_h5 = Path(args.out_h5 or _REPO / "exports" / f"{stem}.h5")
    args.out_pt = Path(args.out_pt or _REPO / "exports" / f"{stem}.pt")
    for path in (args.out_h5, args.out_pt):
        if path.name.startswith("mesh_phase3"):
            raise SystemExit(f"refusing to write {path}: that is the published 36-mode mesh.")
    return args


def build(args):
    torch.manual_seed(args.seed)      # reproducible mesh init
    if args.one_layer:
        return MeshNetwork(args.modes, args.classes, use_svd=True)
    return DeepMeshNetwork(args.modes, args.classes, n_layers=args.layers,
                           alpha=EO_ALPHA, g_phi=EO_G_PHI, phi_b=EO_PHI_B,
                           input_power_w=INPUT_POWER_W)


def _datasets(args):
    train_ds = load_dataset("mnist", subset=args.subset_train or None, split="train")
    test_ds = load_dataset("mnist", subset=args.subset_test or None, split="test")
    return train_ds, test_ds


def deep_parameters(model: DeepMeshNetwork):
    """Handoff-shaped arrays, with the last layer's sigma passivized.

    Every sigma ahead of an activation is already passive (a sigmoid) and is exported
    as the device realises it. Only the last layer's is free, and only there does
    mzi.passivize's argument hold -- its scale cancels in region / total because
    nothing nonlinear follows it.
    """
    thetas, phis, sigmas, outs = [], [], [], []
    gain = 1.0
    for i, layer in enumerate(model.layers):
        sd = {k: v.detach().cpu().double().numpy() for k, v in layer.state_dict().items()}
        sigma = layer.sigma_values().detach().cpu().double().numpy()
        out_v = sd["v.out_phase"]
        if i == len(model.layers) - 1:
            sigma, out_v, gain = mzi.passivize(sigma, out_v)
        thetas.append(np.concatenate([sd["v.theta"], sd["u.theta"]]))     # MESH_ORDER V,U
        phis.append(np.concatenate([sd["v.phi"], sd["u.phi"]]))
        sigmas.append(sigma)
        outs.append(np.stack([out_v, sd["u.out_phase"]]))
    params = {"phase_theta": np.stack(thetas), "phase_phi": np.stack(phis),
              "sigma": np.stack(sigmas), "out_phase": np.stack(outs)}
    return params, gain


def float64_accuracy(params, x_unit, labels, n_classes):
    """The exported model's accuracy, in float64 NumPy -- what MATLAB must reproduce."""
    out, taps = mzi.deep_mesh_forward(params, x_unit, input_power_w=INPUT_POWER_W,
                                      alpha=EO_ALPHA, g_phi=EO_G_PHI, phi_b=EO_PHI_B)
    intensity = np.abs(out) ** 2
    pred = intensity[:, :n_classes].argmax(axis=1)
    return float(np.mean(pred == labels)), taps


def export(model, test_ds, args, torch_acc):
    x_unit = encode_fourier(test_ds.images, n_modes=args.modes).numpy().astype(complex)
    labels = np.asarray(test_ds.labels)
    common = dict(
        geometry={"grid_size": args.modes, "physical_extent_m": 0.0,
                  "n_layers": 2 * (1 if args.one_layer else args.layers),
                  "layer_separations_m": np.zeros(1, dtype="f8")},
        test_images=test_ds.images.astype("f4"),     # 28x28, for display; inputs are data
        test_labels=labels,
        test_inputs=x_unit,
    )
    if args.one_layer:
        sd = {k: v.detach().cpu().double().numpy() for k, v in model.state_dict().items()}
        sigma_p, out_v, gain = mzi.passivize(sd["sigma"], sd["v.out_phase"])
        theta = np.concatenate([sd["v.theta"], sd["u.theta"]])
        phi = np.concatenate([sd["v.phi"], sd["u.phi"]])
        out_phase = np.stack([out_v, sd["u.out_phase"]])
        u = mzi.brick_matrix(sd["u.theta"], sd["u.phi"], sd["u.out_phase"])
        v = mzi.brick_matrix(sd["v.theta"], sd["v.phi"], out_v)
        y = x_unit @ (u @ np.diag(sigma_p.astype(complex)) @ v).T
        acc64 = float(np.mean(np.abs(y[:, :args.classes]).argmax(axis=1) == labels))
        write_handoff(
            args.out_h5, model_type="mesh",
            parameters={"phase_theta": theta, "phase_phi": phi, "sigma": sigma_p,
                        "out_phase": out_phase},
            operating_point={"wavelength_m": args.wavelength, "n_modes": args.modes,
                             "n_classes": args.classes, "readout_gain": model.readout_gain,
                             "sigma_gain": gain, "input_power_w": INPUT_POWER_W,
                             "integration_time_s": SYMBOL_TIME_S},
            description=(f"Phase-5 one-layer baseline | 16 Fourier modes | SVD "
                         f"U*diag(sigma)*V | torch test_acc={torch_acc:.4f} | float64 "
                         f"{acc64:.4f} | seed={args.seed} | sigma passivized, gain "
                         f"{gain:.4f}"),
            test_acc=acc64, **common)
    else:
        params, gain = deep_parameters(model)
        acc64, taps = float64_accuracy(params, x_unit, labels, args.classes)
        write_handoff(
            args.out_h5, model_type="deep_mesh", parameters=params,
            operating_point={
                "wavelength_m": args.wavelength, "n_modes": args.modes,
                "n_classes": args.classes, "readout_gain": model.readout_gain,
                "sigma_gain": gain, "input_power_w": INPUT_POWER_W,
                "integration_time_s": SYMBOL_TIME_S,
                "eo_alpha": EO_ALPHA, "eo_g_phi": EO_G_PHI, "eo_phi_b": EO_PHI_B,
                "eo_tia_gain_ohm": EO_TIA_GAIN_OHM,
                "eo_responsivity_a_per_w": EO_RESPONSIVITY_A_PER_W,
                "eo_v_pi": EO_V_PI, "eo_v_bias": EO_V_BIAS,
                "eo_bandwidth_hz": EO_BANDWIDTH_HZ},
            description=(f"Phase-5 deep mesh | {args.layers} SVD layers, Williamson 2020 EO "
                         f"activation between | 16 Fourier modes | torch test_acc="
                         f"{torch_acc:.4f} | float64 {acc64:.4f} | seed={args.seed} | "
                         f"last sigma passivized, gain {gain:.4f}"),
            test_acc=acc64, **common)
    validate_handoff(args.out_h5)
    return acc64, gain


def main():
    args = parse_args()
    train_ds, test_ds = _datasets(args)
    encoder = lambda imgs: encode_fourier(imgs, n_modes=args.modes)
    model = build(args)
    kind = "one-layer baseline" if args.one_layer else f"{args.layers} layers + activation"
    print(f"Phase-5 mesh | {kind} | modes={args.modes} seed={args.seed} | "
          f"train={len(train_ds)} test={len(test_ds)}")

    if args.export_only:
        if not args.out_pt.exists():
            raise SystemExit(f"checkpoint not found: {args.out_pt}")
        ckpt = torch.load(args.out_pt, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        hist = ckpt["history"]
    else:
        print(f"trainable params: {sum(p.numel() for p in model.parameters())}")
        model, hist = train(model, train_ds, epochs=args.epochs, seed=args.seed, lr=args.lr,
                            batch_size=args.batch, encoder=encoder)
    torch_acc = evaluate(model, test_ds, encoder=encoder, batch_size=500)
    print(f"test accuracy (torch, float32): {torch_acc:.4f}")

    args.out_h5.parent.mkdir(parents=True, exist_ok=True)
    acc64, gain = export(model, test_ds, args, torch_acc)
    if not args.export_only:
        torch.save({"state_dict": model.state_dict(), "history": hist,
                    "args": {k: (str(v) if isinstance(v, Path) else v)
                             for k, v in vars(args).items()}}, args.out_pt)
        print(f"saved torch model -> {args.out_pt}")
    print(f"exported model, float64: {acc64:.4f}  (last sigma passivized, gain {gain:.4f})")
    print(f"exported handoff -> {args.out_h5}  (validated)")


if __name__ == "__main__":
    main()
