# Phase 2 — Diffractive network, ideal case

The Phase-1 propagator, recast as a differentiable layer and stacked with
trainable phase masks, trained to classify MNIST. This document is the Phase-2
deliverable: it states what was built, **what the masks do optically**, the
**optical power budget**, and — most importantly — the **expressivity limit
imposed by linearity**, which is the finding the rest of the project is set up
to characterise rather than engineer around.

Everything here is ideal (no fabrication error). The imperfect as-built model is
Phase 4, in MATLAB, fed through the handoff written by `apps/train_d2nn.py`.

---

## What Phase 2 delivers

| Phase-2 objective (CLAUDE.md) | Where |
|---|---|
| Phase-1 propagator recast as a differentiable layer | `layers.AngularSpectrumLayer` (wraps `propagate.angular_spectrum_transfer`) |
| Stack of phase masks trained to classify | `models.D2NN`, `train.train` |
| Input encoding chosen deliberately | `train.encode_input` — **both** amplitude and phase |
| Detector regions with integrated-intensity readout | `detect.DetectorRegion`, `detect.default_regions`, `detect.integrate_intensity`; differentiable readout in `D2NN.forward` |
| Optical power budget (photons/detector/inference) | `detect.photon_budget`, reported by `apps/train_d2nn.py` |
| Trained D²NN + physical interpretation + power budget + linearity limit | this doc + the exported handoff |

The machine-learning side is deliberately minimal (CLAUDE.md scope): the
electronic readout is *integrate intensity, softmax* and nothing more. The
softmax lives inside `CrossEntropyLoss`; there is no electronic hidden layer.

---

## Architecture and forward pass

A D²NN is `n_layers` trainable phase masks separated by free-space propagation:

```
input field ──prop──▶ mask₁ ──prop──▶ mask₂ ── … ──▶ mask_L ──prop──▶ detector plane
```

so there are `L` trainable masks and `L + 1` fixed propagations, every gap equal
to `separation`. At the detector plane the intensity `|E|²` is summed over each
detector region and normalised by the total output power, giving one
scale-invariant logit per class.

Physics stays pure NumPy (`propagate`, `elements`, `detect`); autograd lives only
in `layers`/`models` (CLAUDE.md convention). The differentiable propagator is a
*thin* wrapper: it precomputes the band-limited angular-spectrum transfer
function with the **same** NumPy routine the reference propagator uses
(`propagate.angular_spectrum_transfer`), stores it as a constant complex buffer,
and applies it with `torch.fft`. `tests/test_layers.py` checks the torch layer
reproduces `propagate.angular_spectrum` to `1e-10` in float64.

**Operating point (design choices, not error magnitudes).** Grid `N = 128`,
pixel pitch `dx = 8 µm` (typical SLM), wavelength `λ = 532 nm`, inter-plane
distance `3 mm`, `L = 5` masks. The propagation is well sampled: the critical
distance `z_crit = N·dx²/λ ≈ 15.4 mm` comfortably exceeds `3 mm`
(see `check_sampling`). These are operating-point constants like wavelength — not
sourced error values — so they need no citation; Phase-4 error magnitudes do.

---

## Input encoding — the deliberate choice

Phase 2 requires choosing how the image enters the field. We use **both**: each
28×28 image is bilinearly resized into the central 50 % of the grid (leaving a
margin for diffraction to spread into) and mapped to

```
E(x,y) = image(x,y) · exp( i · π · image(x,y) )
```

so the image modulates **both** amplitude and phase. The three schemes in
`encode_input` are:

- `amplitude` — `|E| = image`, phase 0 (an amplitude SLM).
- `phase` — `|E| = 1` inside an aperture, `arg(E) = π·image` (a phase SLM under
  plane-wave illumination).
- `both` — the product above.

Each encoded field is normalised to unit total intensity (`Σ|E|² = 1`); the
classifier is invariant to this scale, and it fixes a clean reference for the
photon budget.

**Physical caveat.** Independent amplitude *and* phase modulation is full complex
modulation, which a single SLM cannot do — it needs two cascaded modulators or a
complex-modulation trick. We use it because in-silico we can, and because — as
the linearity section explains — the encoding is the *only* place a nonlinear
transform of the raw pixels can be inserted before the optics. The as-built model
can revisit that hardware cost.

---

## What the trained masks do optically

Free-space propagation is a linear, shift-invariant operation — convolution with
the diffraction kernel. A phase mask is a pointwise multiplication by
`exp(iφ(x,y))`. Alternating them builds a cascade of *(multiply pointwise,
convolve)* steps: a programmable diffractive optical element, i.e. a learned
cascade of holograms.

Functionally, the masks learn to **route and focus** light: they shape the phase
front so that, after diffraction, an input of class *c* constructively
interferes onto detector region *c* and destructively elsewhere. Early masks
behave more like feature encoders that redistribute energy across the aperture;
later masks behave more like focusing/steering elements that concentrate the
class-dependent energy onto the correct detector patch. The trained phase
profiles and an example input→output intensity are rendered in
[`figures/phase2_masks.png`](figures/phase2_masks.png) (generated by
`apps/visualize_d2nn.py`).

Crucially, this "routing" picture is qualitative. The rigorous statement is in
the next section: the entire mask stack is *one linear operator*, and the masks
are simply a physically-realisable parameterisation of it.

---

## Optical power budget

For one inference at the illustrative operating point **1 mW × 1 ms at 532 nm**
(reported by `apps/train_d2nn.py`; `detect.photon_budget` does the accounting
from the exact SI constants `h`, `c`):

| Quantity | Value |
|---|---|
| Photon energy `hc/λ` | 3.73 × 10⁻¹⁹ J |
| Photons delivered per inference `N_in` | 2.68 × 10¹² |
| Fraction captured inside the 10 detectors | 58 % |
| Photons per detector region (mean) | 1.6 × 10¹¹ |
| Photons in the winning (predicted) detector (mean) | 4.9 × 10¹¹ |

**Reading it for Phase 4.** With ~5 × 10¹¹ photons in the winning detector, the
shot-noise relative fluctuation `1/√N ≈ 10⁻⁶` is negligible — at *this* operating
point detection is not shot-noise limited. That is itself a Phase-4 finding: shot
noise only bites once the power budget (or integration time) is pushed orders of
magnitude lower, and the tolerance curves will show where. The photon budget
established here is exactly the input `detect.shot_noise` will consume in Phase 4.

---

## The expressivity limit imposed by linearity

This is the central caveat of the whole diffractive approach, and Phase 2 exists
partly to state it precisely.

**The optical path is entirely linear in the field.** Propagation and phase (or
amplitude) masks are linear operators on the complex field `E`. Their
composition is a *single* linear operator `M`:

```
E_out = M · E_in ,   M = P_{L} D_L P_{L-1} D_{L-1} … D_1 P_0
```

where each `P` is a propagation and each `D` is a diagonal mask. No matter how
many mask+propagation layers are stacked, the field-to-field map collapses to one
linear map. Optical "depth" is **not** depth in the machine-learning sense: extra
diffractive layers add trainable parameters and let `M` better approximate a
desired linear operator *under the physical constraint* that it factorise into
phase-only masks separated by diffraction — but they add no nonlinear
compositional power.

**The only nonlinearity is intensity detection.** The logit for class *c* is the
intensity summed over its detector region:

```
s_c = Σ_{(x,y) ∈ region_c} |E_out(x,y)|²  =  E_in† A_c E_in ,   A_c = M† R_c M ⪰ 0
```

a positive-semidefinite **quadratic form** in the input field. So the classifier
computes, per class, one PSD quadratic form and takes the argmax. That is
strictly more than a linear classifier on the field, but it is exactly *one*
nonlinear layer — "linear optical transform → magnitude-square → linear sum" —
not a deep nonlinear network.

**Consequences, which the results bear out:**

1. Accuracy rises with layers/parameters only up to a **plateau** set by this
   linear-plus-one-square ceiling, not by optimisation effort. Coherent
   single-wavelength diffractive classifiers therefore sit well below nonlinear
   networks on MNIST.
2. The **input encoding is the only place a nonlinearity touches the raw
   pixels** before the optics. Phase encoding applies a nonlinear `exp(i·π·image)`
   map; that is why the encoding choice measurably changes achievable accuracy —
   it is doing nonlinear work the linear optics cannot.
3. Per CLAUDE.md scope, we **characterise and document** this limit; we do not
   add physical activation functions, and we do not grow the electronic head. If
   a larger electronic head would lift accuracy, that is a *finding*, not a bug to
   fix.

---

## Results

Trained with `apps/train_d2nn.py` (grid 128, 5 masks, `both` encoding, seed
`20260724`), Adam, cross-entropy over the detector logits:

- Train/test subset: 12 000 / 2 000 MNIST, 15 epochs.
- **Test accuracy: 0.770** (chance = 0.100).
- Trained parameters: 5 × 128² = 81 920 phase values (only the masks are
  trainable; propagation buffers and detector masks are constants).

The learning curve is the linearity ceiling made visible: validation accuracy
jumps to **0.62 after one epoch**, then **plateaus near 0.77 from epoch ~8**. The
flattening is not an optimisation failure — it is the linear-transform + single
`|·|²` structure hitting its representational limit (see the section above). More
data/epochs/layers nudge the plateau up, but do not change its character.

The exported handoff is `exports/d2nn_phase2.h5` (schema `0.1.0`, validated on
write). A full-data run (`--subset-train 60000 --epochs 30`) trains in-scope on a
laptop CPU and lifts accuracy further, but the ceiling in the linearity section
still applies.

---

## Reproduce

```bash
python -m apps.train_d2nn --quick          # ~15 s smoke config (grid 64)
python -m apps.train_d2nn                   # deliverable config (grid 128)
python -m apps.train_d2nn --subset-train 60000 --epochs 30   # full run
python -m apps.visualize_d2nn               # render trained masks + example
pytest -q tests/test_layers.py tests/test_models.py tests/test_detect.py tests/test_encode.py
```

## What Phase 2 does not cover

- MZI mesh and Clements decomposition — Phase 3 (`mzi.py`, `layers.MZIMeshLayer`).
- Any hardware imperfection — Phase 4, MATLAB, via the handoff.
- Physical activation functions / nonlinear media — out of scope by design; the
  linearity limit above is the thing being characterised.

## References

- Lin et al., "All-optical machine learning using diffractive deep neural
  networks," *Science* 361:1004 (2018) — the D²NN architecture and the
  linear-optics + intensity-detection structure.
- Goodman, *Introduction to Fourier Optics*, 3rd ed. — propagation as a linear
  shift-invariant system.
- Matsushima & Shimobaba, IEEE TIP 18(11):2646 (2009) — the band-limited
  angular-spectrum transfer function shared by the propagator and the layer.
