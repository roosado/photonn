# Phase 5 — an electro-optic activation between mesh layers

Phases 2 to 4 measured a wall. Stacked optics collapse to one linear map, so two meshes
with nothing between them do no better than one, and the only nonlinearity anywhere is
the square-law detector at the end (`docs/phase2_dnn.md`, "The expressivity limit
imposed by linearity"). Phase 5 puts **one** physical nonlinearity between two mesh
layers and asks the project's question of the result:

1. **Does it make depth real?** Does a two-layer activated mesh beat both one mesh and
   two meshes with nothing between them, in this framework, on this task?
2. **What does it cost to build?** The activation's own tolerance edges, and the mesh's
   existing edges re-measured at the new depth.

Question 1 was a gate (plan of record: CLAUDE.md, Phase 5). It passed, on the paper's
input rather than ours. Question 2 is the deliverable.

**The answer, in one paragraph.** On 16 Fourier modes the activation makes a second layer
worth **3.4 ± 0.4 points** (0.8929 against one layer's 0.8598, three seeds), where a
second layer with nothing between is worth nothing. It does so as a pure cubic: no mode
ever reaches the activation's threshold, and the activation bank passes 2.5 × 10⁻⁵ of the
light. So the price is **light, not precision**. The mesh's own tolerances barely move,
but the chip now needs **10 mW to 100 mW** per input at 10 GHz, where one mesh needs
anything above 10 µW, and at the paper's own setting of 1 mW it starves. Every new
precision requirement the activation brings, its bias, its couplers, its amplifier noise
and wavelength drift, relaxes as the power rises, so at the top of the window
they stop binding.

## The device

Williamson, Hughes, Minkov, Bartlett, Pai & Fan, "Reprogrammable electro-optic nonlinear
activation functions for optical neural networks," *IEEE JSTQE* 26(1):7700412 (2020),
doi:10.1109/JSTQE.2019.2930455, arXiv:1903.04579. All equation numbers below are theirs.

A directional coupler taps a fraction α of a mode's power onto a photodiode. The
photocurrent is amplified and drives the phase shifter inside an MZI that the remaining
`1 − α` of the **same** light crosses, after an optical delay that matches the electrical
one. The signal stays optical and coherent: no fresh beam, and nothing digital in the loop.

```
f(z) = j √(1−α) · exp(−j[g_φ|z|² + φ_b]/2) · cos([g_φ|z|² + φ_b]/2) · z      (Eq. 6)

g_φ = π α G ℜ / V_π        phase gain, rad/W                                 (Eq. 7)
φ_b = π V_b / V_π          bias phase                                         (Eq. 5)
```

`G` is the transimpedance gain, `ℜ` the photodiode responsivity, `V_π` the modulator's
half-wave voltage and `V_b` its bias. `|z|²` is power in watts. This is the first place
in the project where absolute optical power is physical rather than a scale that cancels.

**It is our own MZI.** Eq. (6) equals `√(1−α) · [B · P(θ) · B]₁₀ · z` at
`θ = −(φ_b + g_φ|z|²)`, built from `photonn.mzi.beamsplitter` and `phase_shifter`
(`photonn.validate.eo_activation_reference`). Over 2 000 seeded draws reaching ~200 rad
on the modulator, the two agree to **2.6 × 10⁻¹⁴** relative — float64 round-off in `cos`
of a large argument — and `tests/test_eo_activation.py` holds them to 1e-13. That is the
activation's correctness anchor, as Clements reconstruction is the decomposition's.

**Where it operates.** Transmission is `(1−α) cos²((g_φ|z|² + φ_b)/2)`. At φ_b = π and
`g_φ|z|² ≪ 1` it is a pure cubic, `f ≈ −√(1−α)(g_φ/2)|z|²z`, with no linear term. The
paper's MNIST setting — α 0.1, g_φ 0.05π per unit of total input power, φ_b π — sits
there: a mode carrying *all* the light transmits 0.55 %. "ReLU-like" describes the curve's
shape, not where that setting operates.

## The gate

`apps/eo_gate.py`, every arm at three seeds (20260725, 20260726, 20260727). Results in
`exports/phase5_gate/` (gitignored); the numbers below are the record.

### The rule

Registered in plan 12 before anything ran: on the **paper's setup** the activated pair
must beat the linear pair by ≥ 5 points; on **our** 6×6 encoding it must beat one mesh by
more than noise, or Phase 5 switches to the paper's Fourier input; if neither holds, the
phase ends as a null result.

"Noise" was written as the ±0.010 binomial error of 2 000 test images. The first run's
linear arms showed the **seed** spread is larger — one SVD layer scores 0.7355 / 0.7110 /
0.7170 at 20 epochs — so before any activated arm had finished the rule was made
operational: with `Δ` the difference of three-seed means and `SE = √(s₁²/3 + s₂²/3)`,
"beats by more than noise" means **Δ ≥ 0.02 and Δ > 2 SE** (0.05 for the paper rule).

### The protocols

| Protocol | Input | Train / test | Epochs | Batch | Readout |
|---|---|---|---|---|---|
| probe | 6×6 magnitude (`encode_modes`), 36 modes | 20 000 / 2 000 | 5 | 128 | ours |
| ours | same | same | 20 | 128 | ours |
| fourier | 16 complex Fourier modes, unit L2 | 60 000 / 10 000 | 20 | 500 | ours |
| paper | same | same | 200 | 500 | the paper's |

All Adam at 2e-2; "ours" is the shipped mesh's own protocol (`apps/train_mesh.py`). *Our
readout* is the project's "integrate intensity, softmax": `10 · I_k / Σ_all I` as logits.
*The paper's* normalises the ten read intensities by their own sum and uses that as the
probability directly; it also puts an activation after **every** layer, including the last,
in front of the detector. The Fourier input is theirs: the coefficients
`c(kx, ky) = Σ exp(+j kx m + j ky n) g(m, n)` nearest k = 0. Sixteen lands inside the
|k|² = 5 shell, and the paper does not say how it breaks the tie; `train.fourier_order`
uses a stable sort over the shifted grid and says so.

**The script was checked before it was trusted.** One SVD layer under the "ours" protocol
at seed 20260725 scores **0.7355** — the published mesh, reproduced exactly.

### Results (mean ± sd over three seeds)

| Arm | Layers | Activation | Params | Test |
|---|---|---|---|---|
| probe · 1 SVD | 1 | — | 2 628 | 0.6930 ± 0.0280 |
| probe · 2 SVD | 2 | — | 5 256 | 0.6670 ± 0.0048 |
| probe · 2 SVD | 2 | between | 5 256 | 0.6930 ± 0.0200 |
| **ours · 1 SVD** | 1 | — | 2 628 | **0.7212 ± 0.0128** |
| ours · 2 SVD | 2 | — | 5 256 | 0.7045 ± 0.0066 |
| **ours · 2 SVD** (cause 1) | 2 | between | 5 256 | **0.7315 ± 0.0167** |
| ours · 2 unitary | 2 | — | 2 592 | 0.6695 ± 0.0070 |
| ours · 2 unitary (cause 2) | 2 | between | 2 592 | 0.7287 ± 0.0053 |
| ours · 2 SVD, g and φ_b trained (cause 4) | 2 | between | 5 258 | 0.7180 ± 0.0083 |
| **fourier · 1 SVD** | 1 | — | 528 | **0.8598 ± 0.0047** |
| fourier · 2 SVD | 2 | — | 1 056 | 0.8604 ± 0.0068 |
| **fourier · 2 SVD** (cause 3) | 2 | between | 1 056 | **0.8942 ± 0.0057** |
| **fourier · 2 SVD, Σ passive** | 2 | between | 1 056 | **0.8929 ± 0.0056** |
| fourier · 1 unitary | 1 | — | 256 | 0.7158 ± 0.0111 |
| fourier · 2 unitary | 2 | — | 512 | 0.7339 ± 0.0124 |
| fourier · 2 unitary | 2 | between | 512 | 0.8508 ± 0.0078 |
| paper · 1 unitary | 1 | — | 256 | 0.7954 ± 0.0095 |
| paper · 1 unitary | 1 | after | 256 | 0.8610 ± 0.0038 |
| **paper · 2 unitary** | 2 | — | 512 | **0.8066 ± 0.0030** |
| **paper · 2 unitary** | 2 | after each | 512 | **0.8946 ± 0.0031** |

### The verdict, condition by condition

- **The paper's gain reproduces.** Activated pair against linear pair on its setup:
  **+8.8 points** (0.8946 against 0.8066, SE 0.0025), against a bar of 5. The paper
  reports +7.2 (85.83 → 92.98 %, its Table III). The *absolute* levels sit 3 to 5 points
  under the paper's on both sides; the arms share one protocol, which is the only reason
  they may be compared, and the gap between protocols was not chased.
- **On our 6×6 input it does not.** Activated pair against one mesh: **+1.0 ± 1.2**,
  inside noise. The paper's unitary layer (cause 2) gives +0.8 ± 0.8. Training the gain
  and bias (cause 4) gives −0.3: the optimiser pushed g_φ up 7 to 20× (to 1.1 – 3.0 rad
  per unit power), out of the tail toward threshold, and bought nothing with it. Training
  length (cause 1) is what the 20-epoch rows are.
- **So Phase 5 adopts the Fourier input** — the rule's middle exit, outcome B. On that input
  and with **our** readout, the activated pair beats one 16-mode mesh by **+3.4 ± 0.4**
  (0.8942 against 0.8598) and the linear pair by the same, since the linear pair ties one
  layer exactly (0.8604): depth without a nonlinearity is still worth nothing here.
- **It survives the physics.** With a free Σ ahead of the activation, the trained layer
  learned Σ up to 2.85, which is optical gain no chip has. Holding that Σ in (0, 1) — the
  model Phase 5 builds — scores **0.8929 ± 0.0056**, the same.

Two things the table shows that were not the question:

- **An activation in front of the detector is a readout change.** On the paper's setup a
  *single* layer with an activation after it gains +6.6 points (0.7954 → 0.8610): there is
  no second layer for it to sit between. Its effect is to cube the intensities the readout
  normalises. Phase 5's model has no activation after its last layer — the readout stays
  "integrate intensity, softmax", per CLAUDE.md — so its +3.4 is the mid-network
  activation alone.
- **The 6×6 input is the poorer one regardless.** One 16-mode Fourier layer, 528
  parameters, scores 0.8598; one 36-mode 6×6 layer, 2 628 parameters, scores 0.7212. Why
  the activation helps on one input and not the other was not tested. The scratch digital
  probe (plan 12) measured 8 to 10 points of headroom for a nonlinear model on 6×6, so the
  6×6 failure is not a lack of room.

**A trap the first run fell into.** Every activated arm of the first run sat at chance
(0.11 – 0.18). `torch.polar(r, θ)` takes the sign of its result as the direction of
∂/∂r, so its gradient is **wrong in sign** for a negative modulus — and at φ_b = π the
modulus `cos((g_φ|z|² + π)/2)` is negative throughout the weak-light tail. The forward pass
agreed with the reference to 1e-13 either way. `layers.EOActivationLayer` rotates by
`polar(1, −half)` and multiplies by the signed cosine instead;
`tests/test_eo_activation.py` holds it to finite differences, and a canary test fails if
torch ever fixes `polar`.

## The model

`photonn.models.DeepMeshNetwork`: two SVD layers (`U diag(σ) V`, as `MeshNetwork` builds
one) on 16 Fourier modes, with one bank of 16 activations between them and none after.
`MeshNetwork` itself is untouched; the published 0.7355 depends on it.

- **Σ ahead of the activation is passive by construction** (a sigmoid). Only the last
  layer's Σ is free, and it alone is passivized at export (gain 2.30), because
  `mzi.passivize`'s argument — one overall scale cancels in `region / total` — holds only
  where nothing nonlinear follows.
- **Power is physical.** The encoder's unit-norm field enters as `√P_in` times itself; the
  activation's g_φ is in rad/W. The forward pass carries the unit-norm field and writes
  `g_φ · P_in` on it, which is identical in exact arithmetic (Eq. 6 is `c(g|z|²)·z`) and
  avoids the float32 underflow a picowatt field would hit. A test holds the activation's
  input power to scale linearly with `P_in`.
- **Trained** by `apps/train_deep_mesh.py` under the gate's "fourier" protocol, seed
  20260725: **0.8864** on all 10 000 test images, which become the frozen test set.
  Exported to `exports/deep_mesh_phase5.h5` (schema **0.4.0**, model kind `deep_mesh`),
  where the float64 NumPy reference (`mzi.deep_mesh_forward`) scores the exported,
  passivized model at 0.8864 too.
- **The one-layer baseline** is `MeshNetwork(16)` under the same protocol and seed:
  **0.8555** (`exports/mesh16_phase5.h5`, a `mesh` handoff carrying its complex inputs).
  It is the baseline every "change" in the budget is measured from. The published
  36-mode edges are not: different input, different width.
- Both reproduce their gate arms exactly (0.8864 and 0.8555 are seed 20260725's entries
  above), which is the check that the library is the gate's code.

**The as-built anchor.** `photonn-hw/+meshmodel/evaluate_deep.m` reproduces **0.8864
exactly** on all 10 000 images, its logits agree with the float64 reference to
**3.6 × 10⁻¹³** and the power entering every activation to 3.7 × 10⁻¹³ relative.
`meshmodel.evaluate` reproduces the baseline's **0.8555 exactly** (logits to 1.2 × 10⁻¹⁴).

### Where the trained activation operates

`apps/deep_mesh_report.py`, on the frozen 10 000, from the handoff alone, no noise:

| Quantity | Value |
|---|---|
| inputs past half-opening | **0** of 160 000 (image, mode) pairs |
| phase the light writes on the modulator: median / 99th pct / max | 3.9 / 19.5 / 34.9 mrad |
| power reaching the activation bank / input | 0.53 |
| power leaving the bank / power entering it | **2.5 × 10⁻⁵** (−46 dB) |
| power at the readout / input, median image | **1.3 × 10⁻⁶** (−59 dB) |
| photons at the readout per inference, at 1 mW and one 100 ps symbol | **0.98** |

**The gain is a pure cubic.** No mode ever comes near the threshold; the activation that
makes depth real here is `−√(1−α)(g_φ/2)|z|²z`, and the price is light. The bank passes
two parts in a hundred thousand of what enters it.

### The noiseless power sweep

The device fixed as built (g_φ = 157 rad/W), input power swept, every noise source off:

```
P_in (W)    1e-12  1e-9   1e-6   1e-4   1e-3   3e-3   1e-2   3e-2   1e-1   3e-1   1      3      10
accuracy    0.8862 0.8862 0.8862 0.8862 0.8864 0.8863 0.8865 0.8869 0.8795 0.3539 0.2795 0.3136 0.3212
max phase   3e-11  3e-8   3e-5   0.003  0.035  0.10   0.35   1.05   3.49   10.5   34.9   105    349   (rad)
readout/in  1e-24  1e-18  1e-12  1e-8   1.3e-6 1.1e-5 1.3e-4 1.1e-3 1.1e-2 0.041  0.040  0.038  0.038
```

Two regimes, cleanly separated:

- **Below ~30 mW the function does not change.** Twelve decades of input power leave the
  accuracy at 0.8862 – 0.8869, because a pure cubic is scale-covariant under
  `region / total`. What changes is the light: the readout's share falls as `P²`.
- **Above it the function breaks.** At 100 mW the brightest mode writes 3.5 rad and a few
  percent of inputs open past half; by 300 mW the network is at 0.35. That is the
  activation leaving the regime it was trained in. Against this model's own 95 % bar
  (0.8421), the upper edge **holds at 100 mW and fails at 300 mW**.

So the noiseless sweep answers the question plan 12 asked it to: any power edge below
30 mW is **photon starvation, not a change of function**. Only the noisy budget can locate
it.

### The device values

| Quantity | Value | Status |
|---|---|---|
| α, tap fraction | 0.1 | design — Williamson 2020, Fig. 6 |
| φ_b, bias | π | design — Williamson 2020, Fig. 6 |
| g_φ · P_in | 0.05π | design — Williamson 2020, Fig. 6 ("selected heuristically") |
| P_in | 1 mW | design choice — the mesh budget's nominal input |
| ⇒ g_φ | 157.1 rad/W | derived |
| ℜ, responsivity | 1.0 A/W | **measured**: Meyer et al., *Nat. Commun.* 17:3396 (2026), Fig. 2e inset, SiGe photodiode at 3 V, *read from a plot*; also Williamson's Table I design value |
| V_π | 10 V | design example — Williamson 2020, Sec. VII. `UNSOURCED` as a measurement |
| V_b | 10 V | derived (φ_b = π) |
| ⇒ G, transimpedance | 5.0 kΩ (74 dBΩ) | derived from the above. `UNSOURCED`: no datasheet here shows this gain at 10 GHz |
| loop bandwidth | 10 GHz | design — Williamson 2020, Table I ("modulator and detector rate") |
| readout integration | 100 ps | one symbol at that rate |

The ledger with its sources is in `docs/parameter_sources.md`, *The activation*.

## Predictions, registered before the budget

Written down before `photonn-hw/run_error_budget_deep.m` was run, and committed with it
unrun at full scale. One smoke run (2 realizations, 300 images, to check the driver
executes) came between writing them and committing them; no prediction was edited after
it, and the gain sweep's grid was extended past ×101 because that run showed it never
reached a failure. Plan 12 registered the first four; the rest follow from the operating
regime above, which is why they could not have been registered earlier. Each is reported
against what came back in [the budget](#the-as-built-budget).

1. **Serial depth tightens the phase edge** (plan 12). Under the Fourier input the
   comparison is 64 MZI columns against the baseline's 32, not 144 against 72. If error
   accumulates as `σ√depth`, the deep edge is √2 tighter: one step on the sweep grid. If it
   does not move, serial accumulation is not the mechanism.
   *Added:* layer 1's phase errors pass through the cubic, which triples a relative
   amplitude error, so **layer 1 alone should bind harder than layer 2 alone.**
2. **The photon budget stops being free** (plan 12). Now quantitative. The readout's photon
   count goes as `P³T` in the tail (measured: 0.98 photons at 1 mW and 100 ps). Requiring
   the readout count the linear baseline would have at the 36-mode mesh's edge (1 pW @
   1 ms ≈ 155 readout photons) puts the activated chip's **lower edge near 25 µW at 1 ms**,
   7.4 decades above the linear chip, and **near 5.4 mW at 100 ps**, 2.7 decades above the
   linear chip's 10 µW there. With the noiseless upper edge at 100 – 300 mW, the activated
   chip has an **operating window one to two decades wide at 10 GHz**. The linear chip has
   no upper edge at all.
3. **Loss stops cancelling** (plan 12: tighter for layer 1, unchanged for layer 2).
   *Mechanism amended:* uniform loss ahead of the activation is a scale, and the tail is
   scale-covariant, so it cannot move the operating point the way plan 12 said. But the
   mesh's loss is mode-dependent, and the cubic triples a mode-dependent tilt in dB.
   Predicted, noiseless: **layer 1's loss edge about a third of layer 2's**, and layer 2's
   level with the baseline's.
4. **The light cone reaches further** (plan 12). Layer 2's U mesh keeps the closed-form
   dead region: `top − (16 − c + 1) > 10` gives **6** MZIs, the count plan 10 predicted
   for 16 modes. Layer 1 has **none**, because the activation feeds every mode into a full
   second mesh.
5. **The bias edge is set by the signal, in milliradians.** Near φ_b = π a bias error δ
   adds a linear leak `δ/2` against a cubic signal `g_φ|z|²/2`, whose median is 2 mrad.
   Predicted at 1 mW: **holds at ≤ 3 × 10⁻⁴ rad, fails by 3 × 10⁻³** — a hundred times
   tighter than the mesh's own 0.03 rad phase edge — and **30× looser at 30 mW**, because
   the signal grows with power and the leak does not.
6. **The activation's coupler is ~3× tighter than its bias.** An imbalance ε on each
   coupler leaks `ε₂ − ε₁` through the dark state, std `√2 ε`, so it acts like a bias
   error of `2√2 ε`. Same power scaling as 5.
7. **Gain calibration is nearly free.** A common error in g_φ is a common scale in the
   tail and cancels. Since `g_φ` and `P_in` enter only as their product, a gain error of
   (1 + ε) is the noiseless power sweep at `(1 + ε)` mW: **holds to ×30, fails near
   ×100**.
8. **The amplifier's noise is the bias error, per symbol.** An input-referred noise
   density `i_n` at bandwidth B writes a random phase `π G i_n √B / V_π` per symbol —
   1.6 mrad at 10 pA/√Hz and 10 GHz. So the amplifier-noise edge should sit where that
   phase equals prediction 5's bias edge: **near 1 – 2 pA/√Hz** at the design point.

## The as-built budget

Measured after the predictions above were committed (`b41375e`), by
`photonn-hw/run_error_budget_deep.m`. Both machines run through **one protocol**: the
same fixed 2 000-image draw from the frozen 10 000 (ideals 0.8875 deep, 0.8575
baseline), 10 realizations per stochastic magnitude, seeds 29000–50999. Each is judged
against **95 % of its own ideal**: 0.8421 for the activated chip, 0.8127 for the one-layer
chip. Edges are brackets, *holds at X, fails at Y*. Figures in
`photonn-hw/figures_phase5/`, every sweep in `summary.txt` beside them.

### Headline result

| Source | One layer, 16 modes (0.8555) | Two layers + activation (0.8864) | Change |
|---|---|---|---|
| Phase-shifter error | 0.03 / 0.05 rad | **0.03 / 0.05 rad** | **unchanged** |
| … layer 1 only | | 0.02 / 0.03 rad | |
| … layer 2 only | | 0.05 / 0.08 rad | |
| Mesh coupler split | 0.01 / 0.02 | 0.01 / 0.02 | unchanged |
| Thermal crosstalk | 0.002 / 0.005 | 0.002 / 0.005 | unchanged |
| Per-MZI loss | 0.5 / 0.8 dB | **0.3 / 0.5 dB** | one step tighter |
| … layer 1 only | | 0.3 / 0.5 dB | |
| … layer 2 only | | 1.2 / 2.0 dB | |
| DAC resolution | 7 / 6 bits | 6 / 5 bits | within one bit |
| Wavelength drift, at 1 mW | 5 / 10 nm | **2 / 5 nm** | 2.5× tighter: the activation |
| … at 30 mW | | 10 / 20 nm | |
| **Input power, 1 ms symbols** | 1 pW / 0.1 pW | **100 µW / 10 µW** | **8 decades** |
| **Input power, 100 ps symbols** | 10 µW / 1 µW | **10 mW / 3 mW** | **3 decades** |
| **Input power, upper edge** | none (flat to 0.3 W) | **holds 100 mW, fails 300 mW** | **new** |

The activation's own sources, which have no one-layer counterpart:

| Source | At 1 mW | At 30 mW |
|---|---|---|
| Bias-phase error (1-σ) | holds **1 mrad**, fails 3 mrad | holds 30 mrad, fails 100 mrad |
| Activation-MZI coupler split (1-σ) | holds **3 × 10⁻⁴**, fails 1 × 10⁻³ | holds at every value swept (to 0.03) |
| Amplifier input noise, 10 GHz | holds 10 pA/√Hz, fails 30 | holds at every value swept (to 100) |
| Tap-fraction spread (absolute, α = 0.1) | holds 0.02, fails 0.05 | — |
| Phase-gain calibration (multiple of design) | holds from ×0.1 to ×101, fails at ×301 | — |

The light cone: the per-MZI map (θ + 0.5 rad, logit RMS) finds **0, 0, 0 and 6**
MZIs with exactly zero effect in layer 1's V and U and layer 2's V and U, and the six are
the closed form's six, one for one.

### Predictions against measurements

| # | Registered | Measured | |
|---|---|---|---|
| 1 | phase edge √2 tighter at 64 columns than 32 | identical brackets, 0.03 / 0.05 rad | **failed** |
| 1b | layer 1 alone binds harder than layer 2 alone | 0.02 / 0.03 against 0.05 / 0.08 rad; layer 1 carries 70 % of the per-MZI sensitivity | held |
| 2 | lower power edge ~25 µW at 1 ms and ~5.4 mW at 100 ps; a window one to two decades wide at 10 GHz | 10 – 100 µW and 3 – 10 mW brackets, both containing the prediction; window 10 – 100 mW | held |
| 3 | layer 1's loss edge ~⅓ of layer 2's; layer 2 level with the baseline | ¼ (0.3 / 0.5 against 1.2 / 2.0 dB); layer 2 is 2.4× **looser** than the baseline | half held |
| 4 | 6 dead MZIs in layer 2's U mesh, none in layer 1 | 6, matching the closed form exactly; none elsewhere | held |
| 5 | bias holds ≤ 0.3 mrad, fails by 3 mrad at 1 mW; 30× looser at 30 mW | holds 1 mrad, fails 3 mrad; exactly 30× looser at 30 mW | held, conservatively |
| 6 | activation coupler ≈ 2√2 tighter than its bias | 3 × 10⁻⁴ / 10⁻³ against 10⁻³ / 3 × 10⁻³: 3.3× | held |
| 7 | gain error is the power sweep: holds to ×30, fails near ×100 | holds ×101, fails ×301, matching the power sweep point for point | mechanism held; registered bracket one step conservative |
| 8 | amplifier edge where its per-symbol phase equals the bias edge: 1 – 2 pA/√Hz | holds 10, fails 30 pA/√Hz | number **failed** by 10×; mechanism held |

Three of the misses say something:

- **Prediction 1 failed, and that tests plan 10's hypothesis from a second direction.**
  Doubling the serial depth from 32 columns to 64 did not move the phase edge. Serial
  accumulation is real in a linear mesh (`tolerance_mesh.md`, finding 1), but here the
  edge is set by *where* an error sits relative to the nonlinearity: layer 1's errors pass
  through the cubic, layer 2's do not, and layer 2 is the more tolerant of the two by more
  than the √2 its extra columns would cost.
- **Predictions 5, 7 and 8 were conservative for one reason.** The registered brackets
  were read off the full-test-set power sweep, where 100 mW scores 0.8795, and that point
  was taken as failing; against this model's 0.8421 bar it holds. Every mechanism behind
  them held quantitatively (the 30× power scaling of the bias edge, the gain error
  tracking the power sweep exactly, the amplifier noise mapping onto the bias edge: 6 –
  19 pA/√Hz from the measured 1 – 3 mrad). Prediction 8's number inherited prediction
  5's lower bound and was 10× off.
- **Prediction 3's second half failed.** Layer 2's MZIs tolerate 1.2 dB of loss each,
  more than the one-layer chip's 0.5. The light reaching layer 2 is the activation's
  output, already a cube; a mode-dependent tilt after it is not cubed again, and the
  readout depends on it more weakly than on layer 1's.

### Findings

#### 1. Crossing the wall costs light, not precision

Every source the linear chip already had keeps its edge, to within one grid step, except
loss and wavelength, and each of those has a mechanism of its own (findings 4 and 5). The
activated chip needs its phase shifters, couplers and heaters built **no more precisely**
than one mesh does. What the activation adds instead is a power requirement the linear
chip never had: it holds only between 10 mW and 100 mW at its design rate, where the linear
chip holds anywhere above 10 µW.

#### 2. The power edge is photon starvation, and it rises with speed in power but falls in energy

The noiseless sweep showed the function does not change below 30 mW, so the lower edge is
photons alone. The readout's photon count goes as `P³T` in the tail, against `PT` for the
linear chip, and the two symbol times measure the consequence:

| | 1 ms symbols | 100 ps symbols |
|---|---|---|
| one layer, lower edge | 1 pW (**1 fJ** per inference) | 10 µW (**1 fJ**) |
| two layers + activation, lower edge | 100 µW (**100 nJ**) | 10 mW (**1 pJ**) |
| ratio | 10⁸ in power | 10³ in power |

The linear chip's budget is energy per inference and does not care how it is delivered.
The activated chip's is not: delivering the same inference in shorter, brighter pulses
costs **five orders of magnitude less energy**, because the activation responds to power.
It wants exactly the regime Williamson et al. designed it for, fast. And at the setting
they chose, 1 mW at 10 GHz, it is below its own edge: **0.98 photons** reach the readout
per inference, and the chip classifies at 0.18.

The window also has a top. Above 100 mW the brightest modes write radians on their
modulators, the activation leaves the regime it was trained in, and at 300 mW the network
is at 0.35. The linear chip has no such ceiling.

#### 3. The activation's precision is bought with power

At 1 mW the activation is the most demanding component anywhere in this project: its bias
must hold to **1 mrad** (30× the mesh's 0.03 rad phase edge), its MZI's couplers to
**3 × 10⁻⁴** of 50:50 (30× the mesh's 0.01), and its amplifier's noise to 10 pA/√Hz at
10 GHz. All three are errors that leak a *linear* term through a device biased dark, and
they compete with a cubic signal that grows with power. At 30 mW every one of them is 30×
looser: bias 30 mrad, couplers past 0.03, amplifier past 100 pA/√Hz, and wavelength drift
back to 10 nm. **The right operating point is the top of the window**, and there the
activation's tolerances stop binding. What is left is the power bill: 30 mW per input at
10 GHz, three thousand times the linear chip's 10 µW.

#### 4. Depth's cost lands upstream of the nonlinearity

Layer 1 carries 70 % of the network's per-MZI sensitivity (shares 0.38, 0.32, 0.20, 0.10
for layer 1 V and U and layer 2 V and U). Its phase errors fail at 0.03 rad alone; layer
2's hold to 0.05. Its loss fails at 0.5 dB/MZI; layer 2's holds to 1.2. A relative error in
a mode's amplitude ahead of the activation comes out tripled, which is what a cube does, so
a builder spends the precision budget on the first mesh and can relax the second below what
a single linear mesh needs.

#### 5. Wavelength drift becomes an activation problem

At 1 mW the activated chip holds 2 nm where the linear chip holds 5. The control, the
same drift with the activation left at its design wavelength, holds 5 nm, so the
difference is entirely the activation. Its bias sits on a dark fringe, and `φ_b = πV_b/V_π`
moves with `V_π`, which grows with wavelength: 5 nm writes 10 mrad there, common to every
mode. Like the activation's other errors this relaxes with power: at 30 mW the chip holds
10 nm.

#### 6. Gain calibration is free, and it is the one correctable error

A common error in the activation's phase gain, from ×0.1 to ×101, does not move the
accuracy. In the cubic tail a common gain is a common scale and cancels in the readout,
and because `g_φ` and `P_in` only ever appear as their product, a gain error is the
noiseless power sweep run at a different power. The D²NN's calibration gain was the only
correctable error in its budget too (`tolerance_d2nn.md`). Here it is also the one that
does not matter.

#### 7. The light cone survives the activation, on the far side only

Layer 2's U mesh has exactly the six dead MZIs the closed form of
`tolerance_mesh.md` finding 6 predicts at 16 modes, which plan 10 registered as a
prediction for a 16-mode mesh. Layer 1 has none: the activation hands every mode on to a
full second mesh. Six of 480 MZIs (one in 80) need no tolerance at all, against 156 of
the 36-mode mesh's 1 260 (one in eight): a 16-mode mesh read on 10 outputs has little room
outside its cone.

### Required precision per component

To hold 95 % of the ideal 0.8864 (≥ 0.8421), running at 30 mW, one input per 100 ps:

- **Input power: 10 – 100 mW.** Below it the readout starves; above it the activation
  leaves the regime it was trained in. This is the binding constraint, and it is new.
- **Mesh phase shifters ≤ 0.03 rad**, couplers ≤ 0.01, heater coupling ≤ 0.002: what one
  16-mode mesh needs. Layer 2 alone may relax to 0.05 rad.
- **Per-MZI loss ≤ 0.3 dB in layer 1**; layer 2 may take 1.2 dB.
- **Wavelength ≤ 10 nm** (2 nm if the chip were run at 1 mW).
- **Activation bias ≤ 30 mrad, couplers ≤ 0.03, amplifier ≤ 100 pA/√Hz at 10 GHz**, tap
  fraction within ±0.02 of 0.1, phase gain anywhere within ×0.1 – ×100 of design.
- **DAC ≥ 6 bits.**

### Figures (`photonn-hw/figures_phase5/`)

| File | What it shows |
|---|---|
| `tolerance_power.png` | the power window: both machines at both symbol times, and the noiseless curve |
| `tolerance_eo_bias.png` | the activation's bias error at 1 and 30 mW |
| `tolerance_eo_coupler.png` | the activation MZI's coupler imbalance at 1 and 30 mW |
| `tolerance_eo_amp.png` | the activation amplifier's input noise at 1 and 30 mW |
| `tolerance_eo_tap.png`, `tolerance_eo_gain.png` | tap spread; gain calibration |
| `tolerance_phase.png`, `tolerance_loss.png` | both machines, and each layer of the deep one alone |
| `tolerance_coupler.png`, `tolerance_crosstalk.png`, `tolerance_quant.png` | both machines |
| `tolerance_wavelength.png` | both machines, the 30 mW run, and the drift-kept-out-of-the-activation control |
| `sensitivity_map.png` | per-MZI sensitivity, four meshes on the Clements rectangle |
| `confusion_ideal.png` | the as-built deep mesh at its best, 0.8864 |
| `summary.txt`, `error_budget_deep_results.mat` | every sweep |

### Caveats

- **Every device value but one is a design value or `UNSOURCED`.** The responsivity is
  measured (and read from a plot); α, φ_b, the normalised gain, V_π and the 10 GHz rate are
  Williamson et al.'s design values, and the 5 kΩ transimpedance is whatever makes their
  product come out. The edges are properties of this network and this device model; the
  margins against a real photodiode, amplifier and modulator are not in this document.
- **The amplifier is a white input-referred noise density over the loop bandwidth.** No
  bandwidth roll-off, no 1/f noise, no gain-bandwidth limit, no settling: the phase is
  written within one symbol by assumption, as Williamson et al. assume.
- **One input per symbol.** At 1 ms the loop is given a 1 kHz bandwidth to match. A chip
  that held each input for 1 ms but ran its loop at 10 GHz would average many noise draws
  in the readout, which this model does not do.
- **Read noise and ADC are the camera model of the earlier budgets** (2 e⁻, 12 bits,
  Gaussian shot noise). At the photon counts near the 100 ps edge the Gaussian
  approximation to Poisson is rough; it is the same approximation both machines get.
- **The activation's devices are independent.** No crosstalk between neighbouring
  photodiodes or modulators, and the activation's modulator has no heater, so it takes no
  part in the mesh's thermal crosstalk.
- **The mesh sources were measured at 1 mW, the precision list is quoted at 30 mW.** The
  noiseless function is the same at both (0.8864 and 0.8869), so their edges should carry;
  they were not re-measured there. The activation's sources and wavelength were.
- **One seed, one width, one depth.** The gate measured seed spread; the budget is one
  trained model, as the 36-mode budget was.

### Reproduce

```bash
python -m apps.eo_gate --arm all --seeds 20260725 20260726 20260727 --workers 11
python -m apps.train_deep_mesh                # 0.8864 -> exports/deep_mesh_phase5.h5
python -m apps.train_deep_mesh --one-layer    # 0.8555 -> exports/mesh16_phase5.h5
python -m apps.deep_mesh_report               # operating regime, noiseless power sweep
```

```matlab
addpath('photonn-hw')
run_error_budget_deep()                        % about two minutes
run_error_budget_deep(struct('sources', {{'power'}}))   % one source, the rest carried
```

The driver refuses to run unless the as-built model reproduces both handoffs' stated
accuracies exactly, 0.8864 and 0.8555.
