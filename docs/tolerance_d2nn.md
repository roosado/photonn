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

   > **Retracted as a general claim, 2026-08-08.** It holds for *more training*
   > and does not generalise to *more optics*. A 56-mask candidate scoring 0.9040
   > is measurably more fragile on three of six sources — see
   > [A deeper network is a more fragile one](#a-deeper-network-is-a-more-fragile-one)
   > below. The distinction is that the 12k → 60k retrain changed only the fitting,
   > not the geometry; depth changes the geometry.

**Ranking (most to least binding): crosstalk ≫ phase error > detector power ≈
loss (coupled) ≫ wavelength ≈ quantization.** Unchanged from the 12k model.

## A deeper network is a more fragile one

*Measured 2026-08-08 against an unshipped candidate; nothing below describes the
shipped design.* The optics sweep found that spending a fixed diffractive reach
budget on more masks buys a great deal of accuracy (`phase2_dnn.md`), and a
**56-mask, 0.5263 mm** network trained on the full 60 000 for 25 epochs reaches
**0.9040** on the frozen test set against the shipped 5-mask model's 0.7990. The
budget was re-run against it — via `run_error_budget(opts)` with `opts.handoffPath`,
so the shipped handoff was never touched — to ask what that accuracy costs.

Every edge re-measured against the candidate's own stricter bar (0.8588):

| Source | Shipped, 5 masks | Candidate, 56 masks | Change |
|---|---|---|---|
| **Thermal / pixel crosstalk** | holds 0.25 px, fails 0.5 | holds **0.25 px**, fails 0.5 | unchanged — still binding |
| Per-pixel phase error | holds 0.3 rad, fails 0.5 | holds **0.15 rad**, fails 0.2 | **2× tighter** |
| DAC / SLM resolution | holds 3 bits, fails 2 | holds **4 bits**, fails 3 | **1 bit tighter** |
| Optical loss @ knee | holds 1 dB/mask (5 dB total) | holds **0.214 dB/mask** (12 dB total) | **4.7× tighter per mask** |
| Detector / shot noise | holds 1 pW, fails 0.1 pW | holds **0.1 pW**, fails 0.01 pW | 10× looser |
| Wavelength drift | holds 10 nm, fails 20 | holds **20 nm**, fails 30 | 2× looser |

**+10.5 points of accuracy costs 2× phase precision, one more DAC bit and 4.7×
tighter per-element loss.** That is the trade this project exists to quantify, and
it is a more useful result than a clean win would have been.

Three readings worth separating:

1. **The binding constraint does not move.** Crosstalk fails at the same 0.25 px
   edge, and a standard LCoS fringing field (~1 px) destroys either design. Depth
   neither helps nor hurts the thing that already made this unbuildable on an SLM.

2. **The two sources that loosened both follow from photon capture.** The deep
   stack routes **79.1 %** of input photons into the detector boxes against ~60 %
   for the shipped design — the same "route rather than scatter" mechanism the
   accuracy gain comes from. Better SNR at the readout drops the shot-noise knee a
   decade and buys wavelength margin. The design became more robust exactly where
   it already passed by nine orders of magnitude.

3. **Loss points opposite ways in its two units, and the per-element one governs.**
   In total the candidate tolerates *more* attenuation — 12 dB against 5 dB, again
   from the lower knee — but that larger budget is divided among 11× more elements,
   so the per-mask requirement tightens to 0.214 dB. A datasheet quotes per element,
   and 0.214 dB/mask sits at the optimistic end of the 0.2–1 dB realistic range in
   [`parameter_sources.md`](parameter_sources.md). Loss moves from comfortable to
   marginal.

**Required precision for the candidate**, to hold ≥ 0.8588: crosstalk ≤ 0.25 px,
phase ≤ 0.15 rad (λ/42), ≥ 4-bit phase control, ≤ 0.214 dB per mask if operating at
the shot-noise knee, ≥ 0.1 pW × 1 ms, wavelength ≤ 20 nm.

*Not modelled:* `z` falls from 3 mm to 0.53 mm, so a ±10 µm plane-spacing error
goes from 0.33 % to 1.9 % of the gap, and past ~40 masks the stack is better
described as a volume element than as discrete plates. The budget covers device
errors only and has nothing on geometry, so alignment and calibration — still
queued as error sources — plausibly bind this design before anything above does.
**This is flagged, not measured.**

### Two methodological corrections this run forced

- **The loss sweep must scale its range with depth.** It swept 0–10 dB *per mask*,
  a range calibrated for 5 masks. At 56 masks the first non-zero point is 56 dB
  total — a factor of 400 000 in power — so every point sat far below the
  shot-noise knee and read chance. It now spans a fixed 0–30 dB of **total** loss
  and divides by mask count, which is meaningful at any depth. The earlier reading
  of "1 dB/mask" for the candidate was an artifact and is not quoted anywhere.
- **The operating point stayed at 1 pW, and that took checking.** The knee moved
  down to 0.1 pW, and the note in `run_error_budget.m` said to follow it. Following
  it would have been wrong: at 0.1 pW the zero-loss baseline is already 0.8851
  against a 0.8588 bar, so the sweep would measure the knee rather than the loss.
- **The sensitivity map now costs `nMasks × gBlocks²` evaluations** — 36 on the
  shipped design, 2016 at 56 masks, about four hours and ~80 % of the total run.
  `opts.skipSensitivity` reuses a saved map, which is correct exactly when the
  masks are unchanged and only a sweep is being re-measured.

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
