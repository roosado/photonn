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
| Fraction captured inside the 10 detectors | 60 % |
| Photons per detector region (mean) | 1.6 × 10¹¹ |
| Photons in the winning (predicted) detector (mean) | 5.3 × 10¹¹ |

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

- Train/test split: the full **60 000** MNIST training images, 2 000 held-out
  test images, 40 epochs.
- **Test accuracy: 0.799** (chance = 0.100).
- Trained parameters: 5 × 128² = 81 920 phase values (only the masks are
  trainable; propagation buffers and detector masks are constants).

The learning curve is the linearity ceiling made visible. Validation accuracy
reaches **0.750 after a single epoch**, crosses 0.79 by epoch 7, and then climbs
only ~0.01 over the remaining 33 epochs, ending in a 0.796–0.805 band:

| epoch | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 40 |
|---|---|---|---|---|---|---|---|---|
| val | 0.750 | 0.767 | 0.776 | 0.793 | 0.793 | 0.802 | 0.799 | 0.799 |

The decisive number is not the accuracy but the **train/validation gap: 0.798 vs
0.799**. After 40 epochs on 60 000 samples, a model with 81 920 free parameters
still cannot pull ahead on its own training set. It is not data-starved and it is
not under-optimised — loss was still falling monotonically (1.6027 → 1.2725) while
validation had stopped moving. That is the linear-transform + single `|·|²`
structure hitting its representational limit, measured rather than asserted.

**This supersedes an earlier, weaker run** (12 000 images, 15 epochs, 0.7695) that
had not converged. Feeding it the full training set and 40 epochs was worth
**+3.0 points** and cost nothing in parameters, geometry or inference time — but
it bought a plateau, not a trend. Reaching materially higher needs a change to the
optics (more masks, or wider inter-plane separation for more mixing per hop), not
more training. The Phase-4 error budget was re-run against these masks and found
the fabrication tolerances **unchanged** (see `tolerance_d2nn.md`).

The exported handoff is `exports/d2nn_phase2.h5` (schema `0.1.0`, validated on
write).

---

## Seeing the machine: the 3D stage

The trained network runs live in the browser (`apps/web/d2nn.js`, cross-checked to
torch), and the classifier page draws it two ways. The filmstrip is the precise
instrument — seven exact per-plane images. Above it, `apps/web/d2nn_stage.js`
draws the **optical stack itself**: the entrance plane, the five masks and the
detector plane as parallel panels along the optical axis, each carrying the field
computed on it, orbitable, with a sweep that walks one wavefront through.

Three things make that cheap and honest:

- An orthographic projection of a flat plane is **affine**, so one
  `setTransform` + `drawImage` renders a 128² plane as a correct parallelogram —
  no WebGL, no library, consistent with the rest of the browser side.
- The panels are parallel and never intersect, so **back-to-front painting is
  exact occlusion**. (Light is composited additively rather than occluded: the
  masks are transmissive, so a plate must not darken the field behind it.)
- The light **between** the masks is real. Sub-stepping a hop is exact because
  `H(z₁)·H(z₂) = H(z₁+z₂)` and the one z-dependent term — the Matsushima band
  limit — is inactive below `z_crit = N·dx²/λ = 15.40 mm`, against 3 mm hops.
  `NET.sliceForward` computes those intermediate planes; the equality and its
  breakdown above `z_crit` are both asserted in `tests/test_propagate.py`.

Two things it deliberately does not do. It **draws no rays** — scalar diffraction
is not ray optics, and straight lines from digit to detector would misrepresent
the physics this page exists to show. And it **cannot touch the prediction**:
`classify()` still runs the canonical `n_layers+1` propagations that the torch
cross-check pins, and `tests/test_d2nn_crosscheck.py` asserts logits are
bit-identical with slicing enabled.

The stack is 18 mm long across a 1.024 mm aperture — about 18:1 — so drawn to
scale it is an unreadable needle. The depth axis is compressed ~×5.9 and the
figure states that factor on its face.

---

## Reproduce

```bash
python -m apps.train_d2nn --quick          # ~15 s smoke config (grid 64)
python -m apps.train_d2nn                   # 12k subset, 15 epochs (fast, underfits)
python -m apps.train_d2nn --subset-train 60000 --epochs 40   # deliverable (~4.5 h CPU)
python -m apps.visualize_d2nn               # render trained masks + example
python -m apps.d2nn_demo                    # standalone live classifier + 3D stage
pytest -q tests/test_layers.py tests/test_models.py tests/test_detect.py tests/test_encode.py
pytest -q tests/test_d2nn_crosscheck.py tests/test_stage_projection.py
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
