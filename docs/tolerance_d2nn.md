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

Those six are **device** errors: something wrong inside a component. The four below
are **geometry** errors, added 2026-08-17: where the components sit once they exist.
They have no realistic-value column, and that is deliberate — see
[Geometry](#geometry-where-the-parts-sit).

| Source | Tolerance edge (95 % of ideal) | Realistic as-built value |
|---|---|---|
| **Lateral mask registration** | holds at **0.10 px**, fails at **0.15 px** | `UNSOURCED` |
| Detector lateral offset | holds at **2 px**, fails at **3 px** | `UNSOURCED` |
| Plane spacing (per gap) | holds at **100 µm**, fails at **150 µm** | `UNSOURCED` |
| Phase calibration gain | holds at **±10 %**, fails at **±20 %** | `UNSOURCED` |

**Mask registration at 0.10 px is the tightest number in the study**, tighter than
the crosstalk that decides the device half. At the design pitch of 8 µm that is
**0.8 µm per plate**. And the three stochastic geometry sources **do not compose**:
each at the largest magnitude that holds on its own, together they give **0.7119**
against a 0.7591 bar.

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
   > and does not generalise to *more optics*. A 56-mask network scoring 0.9040
   > is measurably more fragile on three of six sources — see
   > [A deeper network is a more fragile one](#a-deeper-network-is-a-more-fragile-one)
   > below. The distinction is that the 12k → 60k retrain changed only the fitting,
   > not the geometry; depth changes the geometry.

**Ranking (most to least binding): crosstalk ≫ phase error > detector power ≈
loss (coupled) ≫ wavelength ≈ quantization.** Unchanged from the 12k model.

## A deeper network is a more fragile one

*Measured 2026-08-08 against the 56-mask network; nothing below describes the
5-mask design.* The optics sweep found that spending a fixed diffractive reach
budget on more masks buys a great deal of accuracy (`phase2_dnn.md`), and a
**56-mask, 0.5263 mm** network trained on the full 60 000 for 25 epochs reaches
**0.9040** on the frozen test set against the 5-mask model's 0.7990. The
budget was re-run against it — via `run_error_budget(opts)` with `opts.handoffPath`,
so the 5-mask handoff was never touched — to ask what that accuracy costs.

Every edge re-measured against the 56-mask network's own stricter bar (0.8588):

| Source | 5 masks | 56 masks | Change |
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

**This table is the caveat the browser carries beside the 56-mask number.** Since
2026-08-09 it runs live on the optics page beside the 5-mask model, on the same
digits — so its number is easy to read as a straight upgrade. It is not one. The
accuracy is real and measured the same way as the headline; what it costs is a
build tolerance this project has no evidence anyone can hit. That cost, not any
status, is what its caption states. See [`phase2_dnn.md`](phase2_dnn.md) for how
it is carried in the bundle rather than written into the page.

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
   In total the 56-mask network tolerates *more* attenuation — 12 dB against 5 dB, again
   from the lower knee — but that larger budget is divided among 11× more elements,
   so the per-mask requirement tightens to 0.214 dB. A datasheet quotes per element,
   and 0.214 dB/mask sits at the optimistic end of the 0.2–1 dB realistic range in
   [`parameter_sources.md`](parameter_sources.md). Loss moves from comfortable to
   marginal.

**Required precision for the 56-mask network**, to hold ≥ 0.8588: crosstalk ≤ 0.25 px,
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
  of "1 dB/mask" for the 56-mask network was an artifact and is not quoted anywhere.
- **The operating point stayed at 1 pW, and that took checking.** The knee moved
  down to 0.1 pW, and the note in `run_error_budget.m` said to follow it. Following
  it would have been wrong: at 0.1 pW the zero-loss baseline is already 0.8851
  against a 0.8588 bar, so the sweep would measure the knee rather than the loss.
- **The sensitivity map now costs `nMasks × gBlocks²` evaluations** — 36 on the
  shipped design, 2016 at 56 masks, about four hours and ~80 % of the total run.
  `opts.skipSensitivity` reuses a saved map, which is correct exactly when the
  masks are unchanged and only a sweep is being re-measured.

## Geometry: where the parts sit

Everything above is a device error. Until 2026-08-17 the study had **nothing
measured** about geometry, and said so in its own voice. For a free-space build that
is the harder half: nobody assembles a five-plane stack with the gaps exactly
3.000 mm and every plate registered to its neighbour. Four sources, measured the
same way as the six above — same handoff, same 95 %-of-0.7990 bar, same
holds-at-X / fails-at-Y bracket convention, seeds 15000–19000 (clear of the device
half's 2000–7000 and the mesh's 8000–14000).

### The measured edges

| Source | Holds | Fails | In units a builder sets |
|---|---|---|---|
| Lateral mask registration, per plate | **0.10 px** | 0.15 px | **0.8 µm** at the 8 µm design pitch |
| Detector lateral offset, whole array | **2 px** | 3 px | **16 µm** |
| Plane spacing, per gap | **100 µm** | 150 µm | 3.3 % of the 3 mm gap |
| Phase calibration gain | **±10 %** | ±20 % | a scale factor, not a physical distance |

### Registration is the tightest requirement in the study

At **0.10 px** it is 2.5× tighter than the crosstalk blur that decides the device
half, and tighter than anything else measured on this design.

The two are not the same quantity — one is the width of a blur, the other the size
of a displacement — so "0.10 against 0.25" is not a like-for-like ratio and should
not be quoted as one. What makes them comparable is the mechanism: both are ways of
getting fine mask structure into the wrong place, and this design's structure is
fine at the pixel scale. `figures/sensitivity_map.png` already showed the masks have
no smooth regions to spare.

The displacements are drawn **independently per plate**, because each plate is
mounted independently. A common displacement of the whole stack would be a much
more benign error: it would translate the output, and the detector layout sits in
the same frame, so most of it would cancel.

### The geometry sources do not compose

Every sweep in this document moves one thing. A real bench has all of them at once,
and the budget had never asked whether its own edges add up. Running the three
stochastic geometry sources simultaneously, each at the largest magnitude that held
alone — spacing 100 µm, registration 0.10 px, detector 2 px — gives

> **0.7119 ± 0.0547**, against the 0.7591 bar. It fails.

**So the per-source edges are not a specification a builder can work to.** Each is
the answer to "how much of this alone", and three sources each at their own limit
are already past the bar together. Any real allocation has to divide the budget, not
spend it three times. This is the first joint measurement in the study; the six
device sources have never been run together either, and on this evidence they should
be.

### A prediction that failed, and what it tells us

`docs/phase3_mesh.md` derives a hard geometric floor: mask separation must stay
above **2.967 mm** against the 3.000 mm design, **33 µm of headroom**, below which
part of the input becomes physically invisible to part of the readout. That was
stated three phases ago and never swept. Per-gap jitter of σ displaces the *mean*
gap by σ/√6, so at the measured 100 µm edge the mean gap is erring by **40.8 µm** —
suspiciously close to 33 µm. The obvious hypothesis is that connectivity is the
mechanism.

It is not. Moving **every gap the same way**, which changes total reach and nothing
else, was swept separately (`tolerance_spacing_systematic.png`):

| systematic offset | −100 µm | −75 | −50 | **−33** | 0 | +50 | +75 | +100 µm |
|---|---|---|---|---|---|---|---|---|
| accuracy | 0.7588 | 0.7725 | 0.7712 | **0.7725** | 0.7712 | 0.7688 | 0.7638 | 0.7512 |

At the connectivity floor accuracy is **0.7725 — indistinguishable from nominal**,
and the curve is roughly symmetric, failing near ±100 µm on both sides. Crossing the
floor costs nothing measurable.

That is not a contradiction; `phase3_mesh.md` predicts it in the same paragraph that
derives the bound: *"the corner of the cone is the Nyquist ray, which carries little
power."* The floor is a statement about what **can** couple, and the measurement
confirms that what stops coupling there carries no useful energy. **A bound the
project had quoted as a tolerance for three phases turns out not to be one.** It
remains correct as geometry, and it is not the operative constraint.

What the random per-gap jitter actually costs, then, is not lost reach — the mean is
well inside the ±100 µm systematic tolerance at every point that holds. It is the
gaps being **unequal**, so the stack no longer matches the geometry its masks were
trained for, plate by plate.

### Phase calibration is loose, and it is the one error you can take back

±10 % holds; even ±20 % only reaches 0.7488. That is remarkable against the device
half's per-pixel phase requirement of 0.3 rad, because the stored phases run to
±23 rad — a 10 % gain error is a *2 rad* error on the largest of them, seven times
the per-pixel tolerance, and it barely registers.

The reason is that the two errors are not the same kind of thing, which is why they
are separate sources. Per-pixel error is zero-mean and independent, so it is
spatially white and scatters light out of the pattern. A gain error is the same
multiplicative bias everywhere at once: it distorts the learned operator coherently
rather than adding noise to it. **Two phase errors of equal size in radians are not
comparable, and a budget that merged them would be wrong.**

The response is also mildly asymmetric — at ±30 %, `k = 1−d` gives 0.7087 and
`k = 1+d` gives 0.7175 — because a trained phase near the 2π wrap scaled up lands on
the far side of it while scaled down it does not. The table quotes the worse sign at
each magnitude, since what a builder needs is a two-sided tolerance.

Gain is also the only source in this study that is **correctable after the fact**: it
is one number, measurable on a test pattern and divided out in software before the
masks are written. The sweep therefore measures how well the calibration must be
*known*, not how well the hardware must behave.

### Why there is no realistic-value column

Every magnitude here is an edge — a property of this network and this geometry, which
no choice of optomechanics will move. What is missing is the other half of a margin:
what a real bench actually achieves for plate registration, stage repeatability and
detector placement. That needs optomechanical measurement literature, and this
project has not sourced it.

Following the precedent set by the mesh budget (`tolerance_mesh.md`), the edges are
published with **every realistic as-built value marked `UNSOURCED` and no margin
column**, rather than filling the gap with plausible numbers. The gap is a table in
[`parameter_sources.md`](parameter_sources.md), not a silence. Until it is closed,
**the device half's verdict — crosstalk fails by 4× — is the only failure this study
can claim**, and the geometry numbers say how hard the build is, not whether it is
possible.

### A conservatism worth stating

The sweeps score an 800-image subset whose ideal accuracy is **0.7713**, while the
bar is 95 % of the full-set 0.7990, i.e. 0.7591. So every sweep starts only 1.2
points above its threshold. All ten sources share this, so they stay comparable to
each other — which is what this study publishes — but each edge is somewhat tighter
than a full-test-set measurement would give. It bites hardest on the geometry
sources, whose curves are flattest near zero.

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

And on where the parts sit:

- **Lateral plate registration:** **≤ 0.10 px = 0.8 µm** per plate, independently.
  The tightest requirement in the study.
- **Detector array placement:** **≤ 2 px = 16 µm** laterally.
- **Plane spacing:** **≤ 100 µm** per gap. Not the 33 µm connectivity floor, which
  is real geometry but costs no measurable accuracy.
- **Phase calibration:** the gain must be *known* to **±10 %**; it need not be
  correct, since it can be divided out before the masks are written.
- **All three together:** the above are single-source edges and **do not compose**.
  A joint allocation must divide the budget between them.

## Figures (`photonn-hw/figures/`)

`tolerance_phase.png`, `tolerance_quant.png`, `tolerance_wavelength.png`,
`tolerance_crosstalk.png`, `tolerance_detector.png`, `tolerance_loss.png`,
`tolerance_registration.png`, `tolerance_detector_offset.png`,
`tolerance_spacing.png`, `tolerance_spacing_systematic.png`,
`tolerance_phase_gain.png`,
`confusion_ideal.png` (no error applied, accuracy 0.7990),
`confusion_phase.png` (as-built confusion at σ = 0.35 rad, accuracy 0.769),
`sensitivity_map.png` (per-mask spatial sensitivity), plus
`error_budget_results.mat`.

**Two confusion matrices, and they answer different questions.** `confusion_ideal.png`
is the model's own behaviour with nothing perturbed — the reference the degraded one is
read against. It is free, since the ideal full-test-set pass is already computed to set
the 95 %-of-ideal threshold. `confusion_phase.png` is what a fabrication error does to
that behaviour.

The stressed matrix's σ is **derived from the model's own measured phase edge** (7/6 of
the largest σ that still holds), not fixed. It reproduces 0.35 rad exactly on the shipped
design, whose edge is 0.3, so that figure is unchanged. On the 56-mask network, whose
edge is 0.15, it gives 0.175 rad and an accuracy of 0.7915. The previous hardcoded
0.35 rad put the 56-mask network nearly twice past its failure point, where it read 0.108 —
chance — and the matrix showed only collapse onto classes 5 and 6.

**The site uses the ideal matrix for both models.** `/` shows the shipped design's
(0.7990) and `/optics` shows the 56-mask network's (0.9040), so the two are directly
comparable and the figure is answering "what did depth buy?" rather than duplicating
the tolerance curves printed beside it. The 56-mask stressed matrix stays in
`figures_candidate_L56/` as a record of the run; it is no longer published.

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
- The geometry sources have **no realistic as-built values**; every one is
  `UNSOURCED`, so the edges below stand alone with no margin (see above).
- Plate **tilt and rotation** are not modelled. Registration covers translation
  only, and a rotated plate is a different (and probably worse) error, since the
  displacement it induces grows with distance from the axis.
- The mask displacements are circular: content leaving one edge of the grid returns
  at the other, the same wrap the propagator already has. At a tenth of a pixel
  against a 128-pixel grid whose outer quarter is dark, this moves no appreciable
  energy.
- Axial detector placement is **not** a separate source: it is the last of the L+1
  gaps and is already inside the spacing sweep. Modelling it twice would quietly
  tighten the joint budget.

## Reproduce

```matlab
cd photonn-hw
addpath(pwd)
run_error_budget                    % full run (~30 min on a laptop CPU)
run_error_budget(struct('quick',true))   % fast reduced run
run_error_budget(struct('handoffPath', '../exports/other.h5'))   % any other model

% One sweep at a time, merging into the saved results. Long background MATLAB
% jobs get killed on this machine and a foreground call is capped, so the full
% run cannot be done in one go; this is how the geometry half was measured.
run_error_budget(struct('sources', {{'registration'}}, 'skipSensitivity', true))
```

Anything `sources` does not name is carried forward from
`figures/error_budget_results.mat`. The ideal baseline is recomputed on every call
regardless and checked against the carried-forward one, so a chunked run still fails
loudly if the model underneath moved.

Deterministic given the recorded seeds; the ideal baseline must read **0.7990** or
the forward model is misaligned with the handoff. `handoffPath` lets a retrained
model be scored without touching `exports/d2nn_phase2.h5`, which is how the 0.7990
masks were checked while that file still held the previous ones.
