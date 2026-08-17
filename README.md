# photonn

Physical photonic neural network simulator. A study of **fabrication tolerance**: train an
idealized optical classifier in-silico, then measure how fast it degrades as real hardware
imperfections are introduced.

> **Central question:** how precisely must a photonic processor be fabricated before it stops
> computing what it was trained to compute?

## Live

### → **[roosado.github.io/photonn](https://roosado.github.io/photonn/)**

**This neural network is made of light.** The trained diffractive network itself, running its
forward pass in your browser: draw a digit or pick one from the frozen MNIST test set, and watch
it diffract through five trained phase masks onto ten detectors. No libraries, no network,
nothing precomputed.

Four more pages read on from it — [the wave optics
underneath](https://roosado.github.io/photonn/physics.html) with the diffraction explorer
recomputing scalar diffraction live as you move the controls, [the same machine built as a
chip](https://roosado.github.io/photonn/chip.html), [how precisely it would have to be
built](https://roosado.github.io/photonn/tolerance.html) — the study, one section per error
source, each with a widget showing what that error physically does — and [how much better the
optics could be](https://roosado.github.io/photonn/optics.html), which measures what depth buys
and then argues why more of it is not the answer.

**The through-line, stated once.** A stack of phase masks is *one linear operator* no matter how
tall it is, so depth converges on the best that operator can do rather than compounding the way
depth does in an ordinary network. Eleven times the masks buys 10.5 points of accuracy and costs
2× tighter phase control, while the one fabrication error that already fails does not move at
all. The way past that ceiling is a nonlinearity, which this project characterises and
deliberately does not try to build.

Two codebases, one project, separated by a one-directional boundary:

- **`photonn/`** — Python. Pure-NumPy scalar wave-optics physics, differentiable (PyTorch)
  models, training. The *ideal design* side.
- **`photonn-hw/`** — MATLAB. Fabrication-error modeling, Monte Carlo, interactive analysis.
  The *as-built* side. Reads the handoff; never writes back.

See [`CLAUDE.md`](CLAUDE.md) for the full architecture, phase roadmap, and scope boundaries.

## Status

**Depth-named models, an honest deep-model matrix, and widgets that hold their shape
(2026-08-14 … 17, `3174e72` → `cd372b1`).** Three passes over what the site claims and how it
renders it.

*A model is named for its depth, never its standing in this project.* The 56-mask network runs
live in the browser, and the page used to label it "not shipped" while a visitor was operating
it — an internal workflow state, and nothing a reader could act on. Bundle labels are now
`5 masks`, `56 masks` and `14 masks`, and each caption states what the accuracy **cost**
instead: *"Accuracy 0.9015 … Buying those points costs 2x tighter phase control and 4.7x lower
loss per mask. Same measurement as 5 masks (0.7995)."* Comparability was never a status
question and still keys on `not_scored_on`, so the 14-mask ranking run still reads *"Not
comparable"*. Nothing measured moved: the tolerance trade below is untouched and the 56-mask
design is **still not promoted**. `provenance.shipped` survives as the guard against two
bundles both claiming to be the headline, but no caption reads it, and
[`tests/test_web_contract.py`](tests/test_web_contract.py)`::test_no_caption_describes_a_model_by_its_status`
scans every committed bundle and fails on any status word.

*The deep model's confusion matrix is drawn at its best, not under stress.* `run_error_budget.m`
now emits `confusion_ideal.png` alongside the stressed one — free, since the ideal full-test-set
pass is already computed to set the 95 %-of-ideal threshold. `/optics` shows the 56-mask
network's at **0.9040** against the front page's **0.7990**, so the two are directly comparable
and the figure answers what depth bought rather than duplicating the six tolerance curves beside
it. Measured from the two matrices: depth repairs the collapse onto 3 (5→3 falls 41 to **6**,
8→3 falls 37 to **11**), which is nearly the whole ten-point gain, while **4 against 9 is not
repaired** (4→9 gets *worse*, 15 to **18**) — a better approximation of the same linear
operation, which is what `/optics` argues next.

*The `/tolerance` widgets hold their shape at any width.* The four before/after/difference
triptychs wrapped on a phone, which grew each panel to full width and left one picture on screen
at a time — no comparison, which is the whole point of them. The two plots drew into a fixed
420-unit space and let `width:100%` stretch the bitmap to the real column, scaling x without y:
flattened on a desktop, stretched tall on a phone. Both fixed, and both faults are now asserted
under Node against a DOM stand-in at three column widths and two device pixel ratios
([`tests/test_error_widgets.py`](tests/test_error_widgets.py)), because the driven browser cannot
see them — that tab is always hidden and never lays anything out.

**Site restructured around the linearity argument (2026-08-14, `360161d`).** `/tolerance` became
one section per error source, each with a purpose-built widget
([`apps/web/errors.js`](apps/web/errors.js)) showing what that error physically does to a real
mask — mechanism only, so it never computes an accuracy and cannot contradict the measured curve
beside it. `/optics` was rebuilt around what depth buys ([`apps/web/scaling.js`](apps/web/scaling.js),
plotting every training run against mask count) and where it stops, closing on the nonlinearity
wall and the routes the field is trying through it. Mathematics is now MathML throughout, which
costs no library. The confusion-matrix stress point is **derived from each model's own measured
phase edge** rather than hardcoded, which fixed a 56-mask figure that had been rendering at
chance because a constant tuned for the 5-mask design is nearly twice past the 56-mask model's
failure point.

**In-browser D²NN classifier (done).** The trained diffractive network now runs its forward pass
client-side on its own page: encode a digit into the entrance field, propagate through the five
trained phase masks, integrate intensity over ten detector regions. It shows the entrance field, the
intensity arriving at each mask, and the detector plane with the class regions drawn on. Trained
parameters are exported to a browser bundle by `python -m apps.export_d2nn_web`
([`apps/web/d2nn_weights.js`](apps/web/d2nn_weights.js), committed — the `.h5`/`.pt` exports are
not versioned). The JavaScript is held to the trained model:
[`tests/test_d2nn_crosscheck.py`](tests/test_d2nn_crosscheck.py) runs it under Node against
reference logits from PyTorch and asserts **identical predictions** (max class-score error 5.5e-7)
plus a bilinear resize matching torch's `align_corners=False` convention to 1.2e-7. Accuracy is
**0.799**, so the shipped gallery deliberately includes digits the model gets wrong.

Above the per-plane frames, the same page draws the **optical stack in 3D**
([`apps/web/d2nn_stage.js`](apps/web/d2nn_stage.js)): entrance plane, five masks and detector plane
as parallel panels along the optical axis, orbitable, with a sweep that walks one wavefront through
and a toggle between the light arriving on each mask and the mask's own trained phase. An
orthographic projection of a flat plane is affine, so each panel is one `ctx.transform` + `drawImage`
— no WebGL, no library — and parallel non-intersecting panels make back-to-front painting exact. The
haze between the panels is the field at **intermediate depths, computed not faked**: sub-stepping a
hop is exact because `H(z₁)·H(z₂) = H(z₁+z₂)` and the band limit is inactive below
`z_crit = 15.40 mm`, asserted in [`tests/test_propagate.py`](tests/test_propagate.py). No rays are
drawn — scalar diffraction is not ray optics — and the slices are display-only: the cross-check
asserts `classify()` logits stay **bit-identical** with slicing enabled.

**Project site + live diffraction explorer (done).** Five self-contained pages tell the whole
story — the working machine first, the fabrication question last — deployed to GitHub Pages by
`python -m apps.build_site`, which writes all of them plus a body-only variant for embedding. The
explorer itself was rebuilt: its
scalar-diffraction physics is ported to ~200 lines of dependency-free JavaScript
([`apps/web/asm.js`](apps/web/asm.js)) — a faithful translation of
`photonn.propagate.angular_spectrum`, cross-checked against it to **< 1e-6**
([`tests/test_asm_crosscheck.py`](tests/test_asm_crosscheck.py)) — so every control (aperture,
distance, wavelength, grid) is continuously live, fixing the previous "only distance is live" caveat.

**Phase 3 — MZI mesh (done).** The interferometer side: a 2×2 MZI transfer matrix from
coupler/phase-shifter primitives (unitary), **Clements and Reck** decomposition of any unitary —
both reconstructing Haar-random unitaries to **~1e-15** — an SVD layer (U·Σ·V†) for arbitrary real
matrices, and a 36-mode mesh classifier ([`MeshNetwork`](photonn/models.py)) trained on
6×6-downsampled MNIST. The headline is a [direct comparison to the D²NN](docs/phase3_mesh.md): the
mesh realises an *arbitrary* linear map with **~31× fewer parameters** (2 628 vs 81 920) and nearly
matches the D²NN (0.74 vs 0.80) — falling just short only because its N²/2-MZI footprint forces
aggressive input downsampling (the footprint ↔ input-dimensionality trade-off). Train/export with `python -m apps.train_mesh`; verify the
decompositions and render the mesh topology with `python -m apps.mesh_toolkit`. The boson-sampling
branch is deferred (open-decision #3).

The same doc also answers **why the two are the same machine**: both are
`[trainable phase] → [fixed mixing] → … → |E|²`, and they differ on one axis — reach per layer.
Diffraction mixes **12.5 px** per hop but unsteerably; a coupler mixes exactly **1 mode** but
individually. That comparison surfaced a result about the D²NN itself: six hops give **74.8 px** of
reach against **74 px** the design needs, so it is fully connected **by 0.81 px (~1%)** — 33 µm of
headroom on the 3 mm mask separation, where the mesh's 36 columns for 36 modes is the Clements bound
and full connectivity is guaranteed by construction. An
[interactive version](https://roosado.github.io/photonn/chip.html) sits on the site's chip page
(`apps/web/analogy.js`); every number in it is read from the trained models by
`python -m apps.export_analogy_web` and re-derived in `tests/test_correspondence.py`.

**Phase 4 error budget on the D²NN (done for the D²NN; MZI sources deferred).** Taking CLAUDE.md
open-decision #2, the fabrication error budget is run against the trained D²NN before the MZI mesh
is built. The MATLAB as-built side (`photonn-hw/`) reproduces the ideal **79.9% baseline exactly**
through its own forward simulator ([`+model`](photonn-hw/+model)), then applies six error sources
([`+err`](photonn-hw/+err): per-pixel phase error, DAC quantization, wavelength drift, thermal
crosstalk, optical loss, detector/shot noise), sweeps each into tolerance curves via Monte Carlo
([`+mc`](photonn-hw/+mc)), and renders confusion matrices and a spatial sensitivity map
([`+viz`](photonn-hw/+viz)). Run the scriptable `run_error_budget` (no App Designer GUI needed).
Findings and required per-component precision are in [`docs/tolerance_d2nn.md`](docs/tolerance_d2nn.md);
every error magnitude is sourced in [`docs/parameter_sources.md`](docs/parameter_sources.md).
Coupler imbalance and per-MZI loss stay Phase-3 stubs (no meaning for phase masks).

**Phase 2 — diffractive network, ideal case (done).** The band-limited angular-spectrum
propagator is recast as a differentiable torch layer (verified against the NumPy reference to
`1e-10`), stacked with trainable phase masks into a [`D2NN`](photonn/models.py), and trained to
classify MNIST. The input is encoded in **both** amplitude and phase; the detector plane reads
integrated intensity over ten regions (*integrate intensity, softmax* — no electronic hidden
layer). The optical power budget (photons per detector per inference) and the **expressivity
limit imposed by linearity** are written up in [`docs/phase2_dnn.md`](docs/phase2_dnn.md). Train
and export with `python -m apps.train_d2nn`; the trained model serialises straight into the HDF5
handoff. MZI meshes are Phase 3; the hardware error model is Phase 4 (MATLAB).

**Phase 1 — wave optics foundation (done).** The scalar-diffraction core: angular spectrum
(band-limited), Fresnel and Fraunhofer, a programmatic sampling criterion, analytic Gaussian-beam
and Airy references, and the aperture/thin-lens elements. The Phase-1 deliverable — a **live, standalone-HTML**
[diffraction explorer](apps/diffraction_explorer.py) that recomputes angular-spectrum propagation
in the browser (see the explainer entry above) — generates from `python -m apps.diffraction_explorer`.

Docs: [`docs/phase3_mesh.md`](docs/phase3_mesh.md) (MZI mesh, decomposition, D²NN comparison) ·
[`docs/tolerance_d2nn.md`](docs/tolerance_d2nn.md) (D²NN error budget, required precision) ·
[`docs/phase2_dnn.md`](docs/phase2_dnn.md) (D²NN, power budget, linearity limit) ·
[`docs/wave_optics.md`](docs/wave_optics.md) (propagators, sampling criteria, validity ranges,
citations) · [`docs/phase0_baseline.md`](docs/phase0_baseline.md) (scaffolding baseline).

## Install

Requires Python 3.11+ (developed on 3.12). From the repo root:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# POSIX:    source .venv/bin/activate
pip install -e ".[dev]"
```

## Test

```bash
pytest -q
```

Physics, layer, model, handoff, site and browser-cross-check tests pass (**`230 passed`**). The
checks that run the browser sources under Node — `test_asm_crosscheck.py`,
`test_d2nn_crosscheck.py`, `test_web_contract.py`, `test_error_widgets.py`,
`test_mount_queue.py` — require Node on `PATH` and skip cleanly if it is absent.

Three things the suite covers that a browser cannot, because the driven Chrome tab here is
always hidden and so never paints or lays out: the widget start-up gate
(`test_mount_queue.py`), the captions rendered from bundle provenance
(`test_web_contract.py`), and the error widgets' layout at several widths and pixel ratios
(`test_error_widgets.py`).

## The handoff

Python writes a single HDF5 file (trained parameters + geometry + operating point + frozen test
set + schema version). MATLAB reads it and models the imperfect device. The contract is specified
in [`docs/handoff_schema.md`](docs/handoff_schema.md); see `photonn/export.py` (writer) and
`photonn-hw/+io/read_handoff.m` (reader).

## Layout

```
photonn/        # Python design side (see CLAUDE.md for per-module responsibilities)
apps/           # diffraction_explorer.py (P1) · train_d2nn.py, visualize_d2nn.py (P2) · train_mesh.py, mesh_toolkit.py (P3) · build_site.py (site) · export_d2nn_web.py, d2nn_demo.py (browser classifier) · export_analogy_web.py, analogy_demo.py, analogy_figure.py (free-space↔chip correspondence)
apps/web/       # dependency-free browser side: asm.js (propagation) · explorer.js (P1 widget) · d2nn.js, d2nn_demo.js, d2nn_stage.js, d2nn_weights.js (trained classifier + 3D stack) · errors.js (P4 error mechanisms) · scaling.js, optics_sweep.js (depth vs accuracy) · d2nn_compare.js (two models, one digit) · analogy.js, analogy_geom.js (P3 correspondence, demo only)
site/           # generated, self-contained, GitHub Pages ready: index.html (the live D²NN) · physics.html · chip.html · tolerance.html (the study) · optics.html
tests/          # pytest suite
docs/           # handoff schema + parameter-source ledger
photonn-hw/     # MATLAB as-built side (+io, +err, +mc, +viz, ErrorBudgetApp)
```
