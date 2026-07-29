# Tolerance document — D²NN error budget

How precisely must the trained diffractive network be built and operated before
it stops computing what it was trained to compute? This is the deliverable of the
error-budget-first branch (CLAUDE.md open-decision #2): the Phase-4 fabrication
error budget run against the Phase-2 D²NN, entirely on the MATLAB as-built side.

## Method

The MATLAB forward simulator (`photonn-hw/+model`) reproduces the Python D²NN's
ideal accuracy **exactly — 0.7695** on the frozen 2 000-image test set from the
handoff (this equality is the correctness anchor). Each error source in
`photonn-hw/+err` is then swept in a Monte Carlo (`+mc/run_montecarlo`, seeds
recorded), accuracy is re-measured, and `run_error_budget.m` renders the tolerance
curves, a confusion matrix, and a spatial sensitivity map.

- **Threshold.** "Retain ≥ 95 % of ideal accuracy" ⇒ accuracy ≥ **0.731**.
- **Sweeps** run on a fixed 800-image subset (subset ideal 0.749) for speed;
  stochastic sources use 10 realizations. Edges below are where the mean crosses
  the threshold and are approximate to the sweep spacing.
- Error magnitudes are sourced in [`parameter_sources.md`](parameter_sources.md).

## Headline result

| Source | Tolerance edge (95 % of ideal) | Realistic as-built value | Margin |
|---|---|---|---|
| **Thermal / pixel crosstalk** | phase blur **σ < ~0.35 px** (≈ 3 µm) | LCoS fringing-field ≈ 1 px | **FAILS** — the binding constraint |
| Per-pixel phase error | **σ < ~0.35 rad** (≈ λ/18) | 0.05 rad (λ/100) … 0.63 (λ/10) | Passes if well-calibrated (λ/100); fails at λ/10 |
| Detector / shot noise | **> ~0.1 pW @ 1 ms** (~tens of photons/detector) | nominal 1 mW (2.7×10¹² photons) | Passes by ~10 orders of magnitude |
| Optical loss | unlimited when photon-rich; **~1 dB** near the shot-noise limit | ~0.2–1 dB per element | Passes at nominal power |
| Wavelength drift | **Δλ < ~15 nm** | TEC-locked ≪ 0.1 nm | Passes by >100× |
| DAC / SLM resolution | **≥ 3 bits** | 8-bit standard | Passes by 5 bits |

## Findings

1. **Spatial phase fidelity is the whole game.** Thermal/pixel crosstalk is by far
   the most damaging source: a Gaussian phase blur of only **σ ≈ 0.5 px** drops
   accuracy from 0.75 to 0.64, and **1 px** collapses it to ~0.19 (near chance).
   The trained masks carry fine, high-spatial-frequency diffractive structure
   (see `docs/figures/phase2_masks.png`); coherent diffraction depends on
   reproducing it precisely, and a standard LCoS SLM's fringing-field crosstalk
   (~1 pixel) would destroy the computation. **This is the tolerance that a real
   build must fight hardest**, and it argues for larger pixels, crosstalk-aware
   (fringing-field-compensated) training, or lithographic phase plates over an
   SLM.

2. **Loss doesn't matter until photons are scarce.** Optical loss cancels exactly
   in the power-normalised readout, so at the nominal photon-rich operating point
   it has **zero** effect on accuracy. It only bites through the photon budget:
   near the shot-noise limit (here 0.1 pW), ~1 dB of insertion loss already
   noticeably degrades accuracy and ~3 dB halves it, because every 3 dB halves the
   photon count. Loss and the power budget must be reasoned about **together**.

3. **The detector is shot-noise limited only far below the design point.** The
   classifier holds accuracy flat from 1 mW down to ~0.1 pW, then falls off a
   cliff as the winning detector drops to a few tens of photons. At the Phase-2
   operating point (1 mW × 1 ms) detection is ~10 orders of magnitude away from
   shot-noise-limited — read noise (2 e⁻) and 12-bit ADC quantization are
   negligible there. Shot noise is a real limit only for very-low-power or
   very-fast operation.

4. **Phase-setting error and quantization are comfortable.** A well-calibrated SLM
   (λ/100 ≈ 0.05 rad) sits well inside the σ < ~0.35 rad tolerance, and even
   3-bit phase control suffices against an 8-bit standard. Wavelength drift is a
   non-issue for any temperature-controlled source (tolerance ~15 nm vs sub-nm
   stability). These are not where a build fails.

**Ranking (most to least binding): crosstalk ≫ phase error > detector power ≈
loss (coupled) ≫ wavelength ≈ quantization.**

## Required precision per component

To hold classification accuracy within 5 % of the ideal 0.77:

- **Inter-pixel phase crosstalk:** keep the effective phase blur **below ~0.35
  pixel** (sub-pixel). This is the hard requirement and likely dictates the
  physical realization.
- **Per-pixel phase calibration:** RMS phase error **< ~0.35 rad** (λ/18); a
  λ/100-class SLM calibration is ample.
- **Optical power / integration:** deliver **> ~a few ×10 photons per detector
  region per inference** (≳ 0.1 pW × 1 ms here); above that, detector noise is
  irrelevant.
- **Insertion loss:** unconstrained when photon-rich; budget **≲ 1 dB total** only
  if operating near the shot-noise limit.
- **Wavelength stability:** **< ~15 nm** (trivially met).
- **Phase DAC resolution:** **≥ 3 bits** (trivially met by 8-bit).

## Figures (`photonn-hw/figures/`)

`tolerance_phase.png`, `tolerance_quant.png`, `tolerance_wavelength.png`,
`tolerance_crosstalk.png`, `tolerance_detector.png`, `tolerance_loss.png`,
`confusion_phase035.png` (as-built confusion at σ = 0.35 rad, accuracy 0.75),
`sensitivity_map.png` (per-mask spatial sensitivity), plus
`error_budget_results.mat`.

## Caveats

- Shot noise uses the Gaussian approximation to Poisson (valid for the large
  counts here; no Statistics Toolbox dependency).
- Optical loss is modelled as uniform amplitude attenuation (it cancels in the
  ideal readout by construction — that is the point of finding #2).
- Per-pixel phase error is modelled as i.i.d.; real errors are spatially
  correlated, so this is conservative. Crosstalk σ-in-pixels is a modelling
  choice (see `parameter_sources.md`).
- Coupler imbalance and per-MZI insertion loss are **not** modelled — they have no
  meaning for phase masks and belong to the Phase-3 MZI mesh.

## Reproduce

```matlab
cd photonn-hw
addpath(pwd)
run_error_budget                    % full run (~13 min on a laptop CPU)
run_error_budget(struct('quick',true))   % fast reduced run
```

Deterministic given the recorded seeds; the ideal baseline must read 0.7695 or
the forward model is misaligned with the handoff.
