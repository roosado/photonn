# Tolerance document — D²NN error budget

How precisely must the trained diffractive network be built and operated before
it stops computing what it was trained to compute? This is the deliverable of the
error-budget-first branch (CLAUDE.md open-decision #2): the Phase-4 fabrication
error budget run against the Phase-2 D²NN, entirely on the MATLAB as-built side.

## Method

The MATLAB forward simulator (`photonn-hw/+model`) reproduces the Python D²NN's
ideal accuracy **exactly — 0.7990** on the frozen 2 000-image test set from the
handoff (this equality is the correctness anchor). Each error source in
`photonn-hw/+err` is then swept in a Monte Carlo (`+mc/run_montecarlo`, seeds
recorded), accuracy is re-measured, and `run_error_budget.m` renders the tolerance
curves, a confusion matrix, and a spatial sensitivity map.

- **Threshold.** "Retain ≥ 95 % of ideal accuracy" ⇒ accuracy ≥ **0.7591**.
- **Sweeps** run on a fixed 800-image subset (subset ideal ≈ 0.771) for speed;
  stochastic sources use 10 realizations. Edges are quoted as the bracket the
  sweep actually resolves — *holds at X, fails at Y* — rather than interpolated
  between grid points, so they can be compared across models without ambiguity.
- Error magnitudes are sourced in [`parameter_sources.md`](parameter_sources.md).

## Headline result

| Source | Tolerance edge (95 % of ideal) | Realistic as-built value | Margin |
|---|---|---|---|
| **Thermal / pixel crosstalk** | holds at **0.25 px**, fails at **0.5 px** | LCoS fringing-field ≈ 1 px | **FAILS** — the binding constraint |
| Per-pixel phase error | holds at **0.3 rad** (λ/21), fails at **0.5 rad** | 0.05 rad (λ/100) … 0.63 (λ/10) | Passes if well-calibrated (λ/100); fails at λ/10 |
| Detector / shot noise | holds at **1 pW @ 1 ms**, fails at **0.1 pW** | nominal 1 mW (2.7×10¹² photons) | Passes by ~9 orders of magnitude |
| Optical loss | irrelevant when photon-rich; at the 1 pW knee holds at **1 dB/mask**, fails at **2 dB** | ~0.2–1 dB per element | Passes at nominal power |
| Wavelength drift | holds at **10 nm**, fails at **20 nm** | TEC-locked ≪ 0.1 nm | Passes by >100× |
| DAC / SLM resolution | holds at **3 bits**, fails at **2 bits** | 8-bit standard | Passes by 5 bits |

## Findings

1. **Spatial phase fidelity is the whole game.** Thermal/pixel crosstalk is by far
   the most damaging source: a Gaussian phase blur of only **σ = 0.5 px** drops
   accuracy from 0.771 to 0.639, and **0.75 px** collapses it to 0.153 — chance is
   0.100, so three quarters of a pixel of blur destroys the computation outright.
   The trained masks carry fine, high-spatial-frequency diffractive structure
   (see `docs/figures/phase2_masks.png`); coherent diffraction depends on
   reproducing it precisely, and a standard LCoS SLM's fringing-field crosstalk
   (~1 pixel) would destroy the computation. **This is the tolerance that a real
   build must fight hardest**, and it argues for larger pixels, crosstalk-aware
   (fringing-field-compensated) training, or lithographic phase plates over an
   SLM.

2. **Loss doesn't matter until photons are scarce.** Optical loss cancels exactly
   in the power-normalised readout, so at the nominal photon-rich operating point
   it has **zero** effect on accuracy. It only bites through the photon budget, so
   it is swept at the shot-noise knee (**1 pW** for this model): 1 dB/mask is still
   fine (0.776), 2 dB crosses the threshold (0.757), 3 dB gives 0.692 and 4 dB
   0.487, because every 3 dB halves the photon count. Loss and the power budget
   must be reasoned about **together**.

   *Methodological note.* This sweep's operating point must sit just **above** the
   detector knee or it measures nothing. It was pinned at 0.1 pW for the previous
   (12k-sample) masks, which was correct for them; the retrained masks moved the
   knee to 1 pW, and left unchanged the sweep began below threshold at 0 dB and
   reported a meaningless "0 dB tolerance". `run_error_budget.m` now carries the
   dependency in a comment. Any future retrain must re-check it.

3. **The detector is shot-noise limited only far below the design point.** The
   classifier holds accuracy flat from 1 mW down to **1 pW**, then falls off a
   cliff: 0.1 pW gives 0.758 (just under the bar), 10 fW 0.487, and 1 fW is chance.
   At the Phase-2 operating point (1 mW × 1 ms) detection is ~9 orders of magnitude
   away from shot-noise-limited — read noise (2 e⁻) and 12-bit ADC quantization are
   negligible there. Shot noise is a real limit only for very-low-power or
   very-fast operation.

4. **Phase-setting error and quantization are comfortable.** A well-calibrated SLM
   (λ/100 ≈ 0.05 rad) sits an order of magnitude inside the 0.3 rad that still
   holds, and even 3-bit phase control suffices against an 8-bit standard.
   Wavelength drift is a non-issue for any temperature-controlled source (10 nm
   holds, against sub-nm stability). These are not where a build fails.

5. **Accuracy went up and tolerance did not go down.** Retraining on the full 60k
   MNIST set for 40 epochs lifted the ideal baseline **0.7695 → 0.7990** (see
   `phase2_dnn.md`), and every tolerance was then re-measured against a
   correspondingly *stricter* bar (0.731 → 0.7591). **Five of the six edges land in
   the same bracket as before** — crosstalk, phase, wavelength, quantisation and
   loss are physically unchanged. Only the detector knee moved, from 0.1 pW to
   1 pW, and that is the tighter threshold biting rather than a real loss of
   sensitivity: 0.1 pW scores 0.7581 against a 0.7591 bar, missing by 0.001.

   The practical reading is that **a better-trained network was not a more fragile
   one** — worth stating because the opposite is a reasonable prior: masks fitted
   to five times more data might have carried finer, more brittle structure. They
   did not.

**Ranking (most to least binding): crosstalk ≫ phase error > detector power ≈
loss (coupled) ≫ wavelength ≈ quantization.** Unchanged from the 12k model.

## Required precision per component

To hold classification accuracy within 5 % of the ideal 0.799 (i.e. ≥ 0.7591):

- **Inter-pixel phase crosstalk:** keep the effective phase blur at or below
  **0.25 pixel**; 0.5 px already fails. This is the hard requirement and likely
  dictates the physical realization.
- **Per-pixel phase calibration:** RMS phase error **≤ 0.3 rad** (λ/21); a
  λ/100-class SLM calibration is ample.
- **Optical power / integration:** deliver **≥ 1 pW × 1 ms**; above that, detector
  noise is irrelevant.
- **Insertion loss:** unconstrained when photon-rich; budget **≤ 1 dB per mask**
  only if operating at the shot-noise knee.
- **Wavelength stability:** **≤ 10 nm** (trivially met).
- **Phase DAC resolution:** **≥ 3 bits** (trivially met by 8-bit).

## Figures (`photonn-hw/figures/`)

`tolerance_phase.png`, `tolerance_quant.png`, `tolerance_wavelength.png`,
`tolerance_crosstalk.png`, `tolerance_detector.png`, `tolerance_loss.png`,
`confusion_phase035.png` (as-built confusion at σ = 0.35 rad, accuracy 0.769),
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
run_error_budget                    % full run (~30 min on a laptop CPU)
run_error_budget(struct('quick',true))   % fast reduced run
run_error_budget(struct('handoffPath', '../exports/other.h5'))   % a candidate model
```

Deterministic given the recorded seeds; the ideal baseline must read **0.7990** or
the forward model is misaligned with the handoff. `handoffPath` lets a retrained
candidate be scored *before* it is promoted, which is how the 0.7990 masks were
checked while `exports/d2nn_phase2.h5` still held the previous ones.
