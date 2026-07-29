# photonn

Physical photonic neural network simulator. A study of **fabrication tolerance**: train an
idealized optical classifier in-silico, then measure how fast it degrades as real hardware
imperfections are introduced.

> **Central question:** how precisely must a photonic processor be fabricated before it stops
> computing what it was trained to compute?

Two codebases, one project, separated by a one-directional boundary:

- **`photonn/`** — Python. Pure-NumPy scalar wave-optics physics, differentiable (PyTorch)
  models, training. The *ideal design* side.
- **`photonn-hw/`** — MATLAB. Fabrication-error modeling, Monte Carlo, interactive analysis.
  The *as-built* side. Reads the handoff; never writes back.

See [`CLAUDE.md`](CLAUDE.md) for the full architecture, phase roadmap, and scope boundaries.

## Status

**Project explainer + live diffraction explorer (done).** A single self-contained
[explainer page](site/index.html) tells the whole story — every phase, all figures, and the Phase-1
diffraction explorer embedded and running **live in the browser** — published as a
[claude.ai Artifact](https://claude.ai/code/artifact/cf11d0f4-09ad-4c36-a30d-a193978c5c71) and
buildable for GitHub Pages (`python -m apps.build_site`). The explorer itself was rebuilt: its
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
matches the D²NN (0.74 vs 0.77) — falling just short only because its N²/2-MZI footprint forces
aggressive input downsampling (the footprint ↔ input-dimensionality trade-off). Train/export with `python -m apps.train_mesh`; verify the
decompositions and render the mesh topology with `python -m apps.mesh_toolkit`. The boson-sampling
branch is deferred (open-decision #3).

**Phase 4 error budget on the D²NN (done for the D²NN; MZI sources deferred).** Taking CLAUDE.md
open-decision #2, the fabrication error budget is run against the trained D²NN before the MZI mesh
is built. The MATLAB as-built side (`photonn-hw/`) reproduces the ideal **77% baseline exactly**
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

Physics, layer, model, handoff, and JS-cross-check tests pass (`61 passed`). The browser-physics
cross-check (`test_asm_crosscheck.py`) requires Node on `PATH`; it skips cleanly if Node is absent.

## The handoff

Python writes a single HDF5 file (trained parameters + geometry + operating point + frozen test
set + schema version). MATLAB reads it and models the imperfect device. The contract is specified
in [`docs/handoff_schema.md`](docs/handoff_schema.md); see `photonn/export.py` (writer) and
`photonn-hw/+io/read_handoff.m` (reader).

## Layout

```
photonn/        # Python design side (see CLAUDE.md for per-module responsibilities)
apps/           # diffraction_explorer.py (P1) · train_d2nn.py, visualize_d2nn.py (P2) · train_mesh.py, mesh_toolkit.py (P3) · build_site.py (explainer page)
apps/web/       # asm.js, explorer.js — the dependency-free live browser-side diffraction explorer
site/           # generated self-contained explainer page (index.html) — GitHub Pages ready
tests/          # pytest suite
docs/           # handoff schema + parameter-source ledger
photonn-hw/     # MATLAB as-built side (+io, +err, +mc, +viz, ErrorBudgetApp)
```
