"""Phase 5: where the trained activation operates, and what input power does to it.

Two design-side measurements plan 12 registered as deliverables, both read from the
handoff alone (through :mod:`photonn.handoff`) and computed in float64 with no noise
of any kind -- noise is the as-built model's job, in MATLAB:

1. **The operating regime.** On the frozen test set, the phase each mode's own light
   writes on its activation's modulator (``g_phi * P``), the fraction of (image, mode)
   inputs past half-opening, and the fraction of the light the activation bank lets
   through.
2. **The noiseless power sweep.** Accuracy against ``input_power_w`` with the device
   fixed as built (``g_phi`` in rad/W does not move). It separates the two ways the
   power edge could arise: the *function* changing (the operating point leaving the
   cubic tail, at high power) and the network starving of *photons* (at low power,
   which only the noisy budget can see). Below the tail the noiseless accuracy cannot
   move, because a pure cubic is scale-covariant under ``region / total``.

Run::

    python -m apps.deep_mesh_report               # prints both, writes JSON
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from photonn import mzi
from photonn.handoff import read_handoff, read_parameters, read_test_inputs

_REPO = Path(__file__).resolve().parent.parent
_PLANCK, _C = 6.62607015e-34, 299792458.0          # exact SI constants

#: Input powers swept, W. The design point is 1 mW; the low end reaches the single
#: mesh's own power edge (1 pW @ 1 ms, docs/tolerance_mesh.md) and the high end runs
#: past the point where the brightest mode would open the activation fully.
POWERS_W = [1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]


def regime(phase, phi_b):
    """Fraction of inputs past half-opening: half the swing from the dark value."""
    t = np.cos(0.5 * (phase + phi_b)) ** 2
    t0 = math.cos(0.5 * phi_b) ** 2
    return float(np.mean(np.abs(t - t0) >= 0.5 * max(1.0 - t0, t0)))


def run(path):
    h = read_handoff(path)
    params = read_parameters(path)
    x = read_test_inputs(path)
    with h5py.File(path, "r") as f:
        labels = f["test_set/labels"][...]
    alpha, g, phi_b = h.op("eo_alpha"), h.op("eo_g_phi"), h.op("eo_phi_b")
    p_design, t_sym = h.op("input_power_w"), h.op("integration_time_s")
    n_classes = int(h.op("n_classes"))
    e_photon = _PLANCK * _C / h.op("wavelength_m")

    def forward(p_in):
        out, taps = mzi.deep_mesh_forward(params, x, input_power_w=p_in, alpha=alpha,
                                          g_phi=g, phi_b=phi_b)
        intensity = np.abs(out) ** 2
        acc = float(np.mean(intensity[:, :n_classes].argmax(axis=1) == labels))
        return acc, intensity, taps

    acc0, intensity, taps = forward(p_design)
    phase = g * taps[0]                                   # rad, per (image, mode)
    p_bank_in = taps[0].sum(axis=1)                       # W entering the bank
    f_bank = mzi.eo_activation(np.sqrt(taps[0]) * 1.0, alpha=alpha, g_phi=g, phi_b=phi_b)
    bank_t = float((np.abs(f_bank) ** 2).sum() / p_bank_in.sum())
    out_power = intensity.sum(axis=1)                     # W at the readout, per image
    report = {
        "handoff": str(path),
        "test_acc_float64": acc0,
        "n_test": int(len(labels)),
        "design": {"input_power_w": p_design, "symbol_time_s": t_sym, "g_phi_rad_per_w": g,
                   "alpha": alpha, "phi_b": phi_b},
        "regime": {
            "phase_median_rad": float(np.median(phase)),
            "phase_p99_rad": float(np.quantile(phase, 0.99)),
            "phase_max_rad": float(phase.max()),
            "frac_past_half_opening": regime(phase, phi_b),
            "bank_power_in_over_input": float(p_bank_in.mean() / p_design),
            "bank_transmission": bank_t,
            "readout_power_over_input_median": float(np.median(out_power) / p_design),
            "readout_photons_per_inference_median": float(
                np.median(out_power) * t_sym / e_photon),
            "readout_photons_in_class_regions_median": float(
                np.median(intensity[:, :n_classes].sum(axis=1)) * t_sym / e_photon),
        },
        "power_sweep": [],
    }
    for p_in in POWERS_W:
        acc, inten, tp = forward(p_in)
        report["power_sweep"].append({
            "input_power_w": p_in,
            "accuracy_noiseless": acc,
            "phase_max_rad": float(g * tp[0].max()),
            "frac_past_half_opening": regime(g * tp[0], phi_b),
            "readout_power_over_input_median": float(np.median(inten.sum(axis=1)) / p_in),
        })
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--handoff", default=str(_REPO / "exports" / "deep_mesh_phase5.h5"))
    ap.add_argument("--out", default=str(_REPO / "exports" / "phase5_report.json"))
    args = ap.parse_args()
    rep = run(Path(args.handoff))
    Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    r = rep["regime"]
    print(f"deep mesh, float64 accuracy {rep['test_acc_float64']:.4f} on {rep['n_test']} images")
    print("operating regime at the design point "
          f"({rep['design']['input_power_w']:.0e} W, g_phi {rep['design']['g_phi_rad_per_w']:.2f} rad/W):")
    for k, v in r.items():
        print(f"  {k:42s} {v:.4g}")
    print("\nnoiseless power sweep (device fixed as built):")
    print(f"  {'P_in (W)':>9s} {'accuracy':>9s} {'max phase':>10s} {'past half':>10s} {'out/in':>10s}")
    for s in rep["power_sweep"]:
        print(f"  {s['input_power_w']:9.0e} {s['accuracy_noiseless']:9.4f} "
              f"{s['phase_max_rad']:10.4g} {s['frac_past_half_opening']:10.4f} "
              f"{s['readout_power_over_input_median']:10.3e}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
