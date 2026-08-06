# photonn

Physical photonic neural network simulator. A study of **fabrication tolerance**: train an
idealized optical classifier in-silico, then measure how fast it degrades as real hardware
imperfections are introduced.

> **Central question:** how precisely must a photonic processor be fabricated before it stops
> computing what it was trained to compute?

## Live

### → **[roosado.github.io/photonn](https://roosado.github.io/photonn/)**

The project explainer: every phase, all figures, and the Phase-1 diffraction explorer recomputing
scalar diffraction **live in your browser** as you move the controls.

### → **[Classify a digit with light](https://roosado.github.io/photonn/classifier.html)**

The trained diffractive network itself, running its forward pass in your browser. Draw a digit or
pick one from the frozen MNIST test set, and watch it diffract through five trained phase masks onto
ten detectors. No libraries, no network, nothing precomputed.

Two codebases, one project, separated by a one-directional boundary:

- **`photonn/`** — Python. Pure-NumPy scalar wave-optics physics, differentiable (PyTorch)
  models, training. The *ideal design* side.
- **`photonn-hw/`** — MATLAB. Fabrication-error modeling, Monte Carlo, interactive analysis.
  The *as-built* side. Reads the handoff; never writes back.

See [`CLAUDE.md`](CLAUDE.md) for the full architecture, phase roadmap, and scope boundaries.

## Status

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

**Project explainer + live diffraction explorer (done).** A single self-contained explainer page
tells the whole story — every phase, all figures, and the Phase-1 diffraction explorer embedded and
running **live in the browser** — deployed to GitHub Pages by `python -m apps.build_site`, which
writes both pages plus a body-only variant for embedding. The explorer itself was rebuilt: its
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
[interactive version](https://roosado.github.io/photonn/#phase3) sits in the Phase-3 section of the site
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

Physics, layer, model, handoff, and JS-cross-check tests pass (`87 passed`). The browser cross-checks
(`test_asm_crosscheck.py`, `test_d2nn_crosscheck.py`) require Node on `PATH`; they skip cleanly if
Node is absent.

## The handoff

Python writes a single HDF5 file (trained parameters + geometry + operating point + frozen test
set + schema version). MATLAB reads it and models the imperfect device. The contract is specified
in [`docs/handoff_schema.md`](docs/handoff_schema.md); see `photonn/export.py` (writer) and
`photonn-hw/+io/read_handoff.m` (reader).

## Layout

```
photonn/        # Python design side (see CLAUDE.md for per-module responsibilities)
apps/           # diffraction_explorer.py (P1) · train_d2nn.py, visualize_d2nn.py (P2) · train_mesh.py, mesh_toolkit.py (P3) · build_site.py (site) · export_d2nn_web.py, d2nn_demo.py (browser classifier) · export_analogy_web.py, analogy_demo.py, analogy_figure.py (free-space↔chip correspondence)
apps/web/       # dependency-free browser side: asm.js (propagation) · explorer.js (P1 widget) · d2nn.js, d2nn_demo.js, d2nn_stage.js, d2nn_weights.js (trained classifier + 3D stack) · analogy.js, analogy_geom.js (P3 correspondence)
site/           # generated, self-contained, GitHub Pages ready: index.html (explainer) · classifier.html (live D²NN)
tests/          # pytest suite
docs/           # handoff schema + parameter-source ledger
photonn-hw/     # MATLAB as-built side (+io, +err, +mc, +viz, ErrorBudgetApp)
```
