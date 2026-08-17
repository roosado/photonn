# Tolerance document — MZI mesh error budget

How precisely would the Phase-3 interferometer mesh have to be fabricated before it stops
computing what it was trained to compute? This is the Phase-4 fabrication error budget run
against the MZI mesh, entirely on the MATLAB as-built side — the companion to
[`tolerance_d2nn.md`](tolerance_d2nn.md), which did the same for the diffractive network.

It exists to answer a question the D²NN budget could not. That study modelled six sources
and left two out, because a phase mask has no coupler and its loss is uniform
(`tolerance_d2nn.md`, Caveats). Those two are the mesh's own, and one of them turns out to
matter more than anything the D²NN could have told us.

## Method

Same apparatus, same discipline, one architecture over:

- **The correctness anchor.** `photonn-hw/+meshmodel` reproduces the Python accuracy
  **exactly, 0.7355**, on the frozen 2 000-image test set. Its logits agree with the NumPy
  reference to **8 × 10⁻¹⁵** across all 2 000 × 10 — float64 round-off, not approximation.
  That match is what makes every number below a measurement of the error model rather than
  of a transcription mistake.
- **The machinery.** `photonn-hw/+err` swept by `+mc/run_montecarlo_mesh` (seeds recorded),
  driven by `run_error_budget_mesh.m`, figures in `photonn-hw/figures_mesh/`.
- **The threshold:** retain ≥ 95 % of ideal ⇒ **≥ 0.6987**. Model-relative, exactly as the
  D²NN's 0.7591 is relative to its own 0.7990.
- **The protocol:** 800-image subset for the sweeps (subset ideal **0.7250**), 10
  realizations for stochastic sources, and edges quoted as a **bracket** — "holds at X,
  fails at Y" — never interpolated, so they compare across models without ambiguity.
- **The sensitivity map** runs on the whole 2 000. It costs about a minute here; the
  D²NN's cost four hours and had to be subsetted hard.

**Every realistic as-built value in this document is `UNSOURCED`, deliberately and in that
order.** The D²NN's device is a visible-wavelength phase-mask processor and its parameters
came from the SLM and sCMOS literature; a silicon mesh needs a PDK-anchored set that this
project does not yet have (`CLAUDE.md` open decision #4, and the mesh block in
[`parameter_sources.md`](parameter_sources.md)). Rather than invent numbers, the study
measures the **edges** — which are properties of the trained network and its topology, not
of any foundry — and leaves the comparison column empty. So there is no "Margin" column
here. Adding one would be fiction.

## Headline result

| Source | Tolerance edge (95 % of ideal) | Realistic as-built value |
|---|---|---|
| **Phase-shifter error** | holds at **0.03 rad** (λ/209), fails at **0.05 rad** | `UNSOURCED` — thermo-optic setting accuracy |
| **Coupler imbalance** | holds at **0.01**, fails at **0.02** power-split σ | `UNSOURCED` — MMI / directional-coupler process spread |
| **Thermal crosstalk** | holds at **0.005**, fails at **0.01** coupling coeff. | `UNSOURCED` — heater pitch and thermal decay length |
| Wavelength drift | holds at **5 nm**, fails at **10 nm** | `UNSOURCED` — coupler dispersion dominates; see finding 4 |
| Per-MZI insertion loss | holds at **0.5 dB/MZI**, fails at **0.8 dB** | `UNSOURCED` — per-MZI insertion + waveguide loss |
| DAC resolution | holds at **6 bits**, fails at **5 bits** | 8-bit standard (electronics, architecture-independent) |
| Detector / shot noise | holds at **1 pW @ 1 ms**, fails at **0.1 pW** | nominal 1 mW — passes by ~9 orders |

Ranked: `phase error ≈ coupler imbalance ≈ crosstalk ≫ wavelength > loss > quantization ≫
detector power`. The top three are within a factor of a few of each other in relative terms,
which is itself the story — see finding 2.

## Findings

### 1. Serial depth is what costs the mesh its phase tolerance

**The mesh needs its phases ten times more accurate than the D²NN does.** It holds at
**0.03 rad** where the D²NN holds at **0.3 rad** (λ/209 against λ/21). The degradation
series on the 800-image subset, against its 0.7250 ideal:

```
sigma (rad)   0     0.005  0.01   0.02   0.03   0.05   0.08   0.12
accuracy      0.7250 0.7236 0.7204 0.7135 0.7145 0.6838 0.5835 0.4265
```

This is the prediction `docs/phase3_mesh.md` wrote down before anything was measured —
*"the mesh's error is dominated by accumulation through 72 serial MZIs"* — and it holds. A
D²NN phase error is a **parallel** insult: 16 384 pixels each perturbed once, and the
detector integrates over all of them, so the errors average. A mesh phase error is
**serial**: light crosses 36 columns, and an error in column 3 is re-mixed by every column
after it. Averaging helps the first architecture and compounds in the second.

The two are directly comparable because both are a 1-σ Gaussian in radians on a programmed
phase — `err.phase_shifter_error` is literally the same function, dispatching on which
parameter set it was handed.

### 2. Splitting error and phase error are the same size, and that is the real result

Sweeping θ and φ separately at the same σ:

```
sigma (rad)   0      0.005  0.01   0.02   0.03   0.05   0.08   0.12
theta only    0.7250 0.7220 0.7226 0.7186 0.7134 0.7011 0.6403 0.5734
phi only      0.7250 0.7235 0.7226 0.7226 0.7215 0.7088 0.6756 0.6235
both          0.7250 0.7236 0.7204 0.7135 0.7145 0.6838 0.5835 0.4265
```

θ, which sets an MZI's splitting ratio, is consistently worse than φ, which only shifts a
phase — as expected. But the gap is small, and **both alone hold to 0.05 rad while the two
together fail there.** The errors add rather than one dominating, so there is no single
shifter to spend the calibration budget on. Both phases per MZI need the same accuracy.

### 3. Coupler imbalance is a first-class constraint, and it is new

The source the D²NN budget could not model at all sits **level with phase error** in
severity: 0.01 holds, 0.02 fails, and by 0.05 the network is at 0.5936.

```
split sigma   0      0.005  0.01   0.02   0.03   0.05   0.08   0.12
accuracy      0.7250 0.7225 0.7143 0.6983 0.6703 0.5936 0.3739 0.2335
```

Why it bites where a phase error partly does not: an MZI's splitting is set by θ **given**
50:50 couplers. Move a coupler off 50:50 and the MZI can no longer reach full extinction at
*any* θ — the error is not a phase a later stage can partly undo, it is a floor on how well
that MZI can route. A mesh calibrated in-situ could trim θ to compensate in part; this
study is in-silico training then transfer, which is the project's entire premise, so it
cannot.

### 4. Wavelength drift is a coupler problem, not a phase problem

The mesh holds to **5 nm** against the D²NN's 10 nm — but the interesting number is the
control. Rescaling the phases alone (`λ₀/λ` through the shifter path length, no coupler
dispersion) holds to **20 nm**:

```
drift (nm)          0      1      2      5      10     20     30
both effects        0.7250 0.7250 0.7225 0.7100 0.6888 0.6238 0.5650
phase rescale only  0.7250 0.7238 0.7275 0.7288 0.7188 0.7088 0.6900
```

So essentially all of the failure is the couplers drifting off 50:50, not the shifters
drifting off their setting. **Wavelength drift is a systematic coupler imbalance** — applied
to every coupler in the same direction rather than drawn per device — which makes source 4
and source 3 the same physics at different correlations. The findings say so rather than
presenting them as six independent sources and implying they are.

### 5. Loss stops being free

In the D²NN, loss is a near-uniform attenuation that **cancels** in the power-normalised
readout: it had to be measured at the 1 pW photon knee to bite at all. In a Clements
rectangle it does not cancel. The mesh is a brick, so a mode near the edge crosses fewer
MZIs than one in the middle, and per-MZI loss therefore **tilts** the realised linear map
rather than scaling it. Measured with an ideal detector — no photon budget involved — it
holds at 0.5 dB/MZI and fails at 0.8.

That is a much larger per-element number than the D²NN's 1 dB/mask looks next to, and the
comparison is a trap: a mode here crosses about 36 MZIs, so 0.5 dB/MZI is ~18 dB across the
mesh against the D²NN's 5 dB across five masks. The mesh tolerates more *total* loss and
much less *per-element* loss, and neither statement alone is the finding.

### 6. A quarter of the U mesh does not have to be built well

The per-MZI map (Fig 8) detunes one MZI's θ by 0.5 rad and measures how far the class logits
move. It is not uniform, and the structure is exact rather than statistical.

**156 of the U mesh's 630 MZIs — 25 % — have precisely zero effect.** Not "small": zero. The
readout reads the first 10 of 36 modes, an MZI in column *c* can influence a mode at most
`36 − c + 1` hops away, and every MZI failing `top − (36 − c + 1) ≤ 10` lies outside the
readout's light cone. That geometric criterion partitions the mesh with **no errors in
either direction**: the 474 inside carry 100.0 % of the sensitivity, the 156 outside carry
0.0 %.

Concentration is steep even inside: half of the U mesh's total sensitivity sits in **93
MZIs (15 %)**, and half of the V mesh's in **159 (25 %)**. V is broader, as it should be —
it mixes the input into all 36 modes, so it has no dead cone on the readout side.

This is the most directly actionable thing in the document. Tolerance is usually quoted per
device and applied uniformly; here a quarter of one mesh could be built to no tolerance at
all, and the boundary is known in closed form before anything is fabricated.

### 7. Detector power and quantization are architecture-independent, and confirm each other

The detector edge is **1 pW at 1 ms**, failing at 0.1 pW — the *same* edge the D²NN
measured, on a different architecture, a different test set and a different wavelength. It
should be: it is a photon-counting statement about the readout, not about the optics in
front of it. Getting the same number twice is a check on both budgets.

Quantization holds to **6 bits** against the D²NN's 3 — three bits tighter, the same
serial-accumulation story as finding 1, and still met with room by an 8-bit driver.

## Against the D²NN

The comparison this study exists for. **Both columns are each model's own 95 %-of-ideal
edge**, which is the only way to put them side by side.

| Source | D²NN, 5 masks (0.7990) | Mesh, 36 modes (0.7355) | Change |
|---|---|---|---|
| Phase error | holds 0.3 rad (λ/21) | holds **0.03 rad** (λ/209) | **10× tighter** |
| DAC resolution | holds 3 bits | holds **6 bits** | **3 bits tighter** |
| Wavelength drift | holds 10 nm | holds **5 nm** | 2× tighter — and it is the couplers |
| Detector power | holds 1 pW @ 1 ms | holds **1 pW @ 1 ms** | **unchanged** |
| Loss | 1 dB/mask (5 dB total) | **0.5 dB/MZI** (~18 dB total) | tighter per element, looser in total |
| Crosstalk | 0.25 px blur | 0.005 coupling coeff. | not comparable — different mechanism, different units |
| Coupler imbalance | *no meaning* | holds **0.01** power split | **new**, and level with phase error |

**Read it as: the mesh buys an arbitrary linear map with 31× fewer parameters, and pays for
it in precision.** The D²NN's parameters are redundant — 16 384 pixels expressing a
constrained map — and redundancy is exactly what makes a device tolerant. The mesh has no
redundancy: 2 628 parameters realising a map that genuinely needs 2 628 numbers, arranged so
each one is seen by everything downstream. The same property that makes it efficient makes
it fragile, and that is one trade, not two facts.

**Nothing here says which architecture is better.** The D²NN's binding constraint (pixel
crosstalk at 0.25 px against ≈1 px of real LCoS fringing) **fails against real hardware**
today. The mesh's binding constraints are unsourced, so whether 0.03 rad and a 0.01 split
tolerance are comfortable or hopeless in a real process is exactly the question this study
cannot yet answer. What it establishes is where the edges are.

**The caveat that governs the whole table:** the two models run on **different frozen test
sets** — 28×28 in a 128² field against MNIST downsampled to 6×6 — at different ideals
(0.7990 against 0.7355). Only the edges compare, each against its own bar. **Absolute
accuracies must never be compared**, and the 0.7355 is not evidence that a mesh classifies
worse than a diffractive stack; it is evidence that 36 modes is a small input.

## Required precision per component

To hold classification accuracy within 5 % of the ideal 0.7355 (i.e. ≥ 0.6987):

- **Phase-shifter setting:** RMS error **≤ 0.03 rad** (λ/209), on **both** θ and φ. This
  is the hard requirement and it is ten times what the diffractive design needed.
- **Coupler split:** **≤ 0.01** 1-σ deviation from 50:50 in power. Equally hard, and not
  fixable by trimming θ under in-silico training.
- **Heater thermal coupling:** **≤ 0.005** coefficient at the modelled layout. Sensitive to
  a pitch and a decay length that are not yet sourced, so treat the ranking, not the value.
- **Wavelength stability:** **≤ 5 nm**, and the requirement is on the *couplers'* bandwidth,
  not the shifters'.
- **Insertion loss:** **≤ 0.5 dB per MZI**.
- **Phase DAC resolution:** **≥ 6 bits** (met by 8-bit).
- **Optical power / integration:** **≥ 1 pW × 1 ms**.
- **Not uniformly:** 25 % of the U mesh is outside the readout's light cone and needs none
  of the above. See finding 6.

## Figures (`photonn-hw/figures_mesh/`)

| File | What it shows |
|---|---|
| `confusion_ideal.png` | the as-built model at its best — 0.7355, matching Python exactly |
| `confusion_phase.png` | the same model stressed past its phase edge — 0.035 rad ⇒ 0.7100 |
| `tolerance_phase.png` | phase-shifter error, the joint θ+φ sweep |
| `tolerance_coupler.png` | coupler imbalance — the source the D²NN could not model |
| `tolerance_crosstalk.png` | thermal crosstalk through the distance-dependent matrix |
| `tolerance_loss.png` | per-MZI insertion loss, measured without the photon budget |
| `tolerance_wavelength.png` | wavelength drift, phases and couplers together |
| `tolerance_quant.png` | DAC resolution |
| `tolerance_detector.png` | photon budget and detector noise |
| `sensitivity_map.png` | per-MZI sensitivity on the Clements rectangle, both meshes |
| `error_budget_mesh_results.mat` | every sweep, plus both sensitivity maps |

Two notes on the figures.

**The stressed confusion matrix's σ is derived from this model's own measured phase edge**
(7/6 of the largest σ that still holds ⇒ 0.035 rad ⇒ 0.7100), never hardcoded. A constant
tuned for one model renders another at chance and reads as a broken figure rather than the
fragility result it is; the D²NN study learned that the hard way.

**The sensitivity map plots the RMS logit shift, not the accuracy drop**, and the D²NN's
plots accuracy. The reason is resolution, not preference. The D²NN perturbs a 21×21 block of
a mask — a large kick, and accuracy moves clear of its own granularity. One MZI out of 1 260
does not: accuracy on 2 000 images moves in steps of 0.0005, a single MZI flips a handful of
images either way, and measured that way 80 % of the mesh reads exactly zero with a
*negative* mean. That is quantisation noise, not physics. The logit RMS measures the same
disturbance with no floor under it. Both maps are in the `.mat`.

`viz.mesh_sensitivity_map` is also a different renderer, because `viz.sensitivity_map` lays
out 1×L subplots at 260·L px — which for the 56-mask D²NN produced a 16 139 × 341 plate at
47:1, unpublishable at any web size. A mesh map is two panels of 36 columns by 18 rows and
is near-square by construction.

## Caveats

- **No realistic as-built value is sourced.** Stated at the top, restated here: the edges
  are measured, the comparison column is not. This is the largest open item in the study.
- **The thermal model's geometry is invented.** The decay length (50 µm) and heater pitch
  (80 × 40 µm) are fixed constants in `run_error_budget_mesh.m`, both marked `% UNSOURCED`.
  The sweep is over the coupling coefficient alone, so the curve's x-axis is the one
  quantity a measurement would pin down — but the whole curve moves if the geometry does.
- **The exponential coupling kernel is a modelling choice**, the standard lumped form for
  in-plane heat spreading, not a solution of the heat equation for this layout.
- **V and U are treated as thermally independent.** Two rectangles with the Σ bank between
  them. That is the optimistic assumption; a chip that packs them together would couple them.
- **Loss is deterministic per MZI.** Process mean only — run-to-run spread in loss is a
  separate draw and is not modelled.
- **Σ is realised as an ideal passive attenuator bank.** The trained diagonal was signed and
  ran to 3.907; `photonn.mzi.passivize` folds the sign into V's output phase and the scale
  into an external gain of **3.9068**, both provably logit-preserving. What is *not*
  modelled is error in the attenuators themselves.
- **In-silico training then transfer**, per the project premise. A mesh calibrated in-situ
  could trim θ against a measured coupler imbalance and would show different numbers. That
  is a different experiment, not a correction to this one.
- **One mesh size.** 36 modes, one point. Whether the phase edge scales as 1/√L or 1/L with
  mode count is the obvious follow-on and would need a second trained mesh.

## Reproduce

```matlab
cd photonn-hw
addpath(pwd)

run_error_budget_mesh()                              % full run, minutes
run_error_budget_mesh(struct('quick', true))         % smoke config
run_error_budget_mesh(struct('skipSensitivity', true))
```

The ideal baseline must read **0.7355** or the forward model is misaligned with the handoff.
That number comes from `exports/mesh_phase3.h5` at schema **0.2.0**; a 0.1.0 mesh handoff is
missing Σ and the output phases and `io.read_handoff` will refuse it. Re-export with
`python -m apps.train_mesh --export-only`, which rebuilds the file from the checkpoint
without retraining, so the anchor cannot drift.
