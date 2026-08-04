# photonn — Project Context

Physical photonic neural network simulator. Two codebases, one project.

- `photonn/` — Python. Physics simulation, differentiable models, training.
- `photonn-hw/` — MATLAB. Hardware error modeling, Monte Carlo, interactive analysis.

**Central question the project answers:** how precisely must a photonic processor be
fabricated before it stops computing what it was trained to compute?

The machine learning content is intentionally minimal. Do not expand it. All complexity
belongs in the optical physics and the hardware error model.

---

## Environment

- Laptop-only. No cluster, no GPU assumed. Anything that requires either is out of scope.
- Python: NumPy, SciPy, PyTorch (or JAX — see open decisions), Matplotlib, Plotly, h5py.
- MATLAB: base + App Designer. No specialized toolboxes should be required; flag it if one becomes necessary.
- Target field resolutions: 128² during development, 512² maximum. Do not exceed 512² without an explicit reason — memory during backprop becomes the binding constraint.

---

## Architecture

```
photonn/
├── fields.py         # complex field objects: physical extent, sampling, wavelength, units
├── propagate.py      # angular spectrum, Fresnel, Fraunhofer; sampling validators
├── elements.py       # phase masks, amplitude masks, lenses, apertures
├── mzi.py            # 2x2 MZI transfer matrix, Clements/Reck decomposition, mesh forward pass
├── layers.py         # differentiable (torch/jax) wrappers around propagate/elements/mzi
├── models.py         # D2NN and mesh network definitions
├── train.py          # training loops, datasets, input encoding schemes
├── detect.py         # detector regions, photon budget, shot and thermal noise
├── export.py         # serialize trained parameters for MATLAB handoff
└── validate.py       # analytic test cases and invariant checks

photonn-hw/
├── +io/              # load exported parameter sets and frozen test data
├── +err/             # fabrication error, quantization, dispersion, loss, thermal crosstalk
├── +mc/              # Monte Carlo drivers and statistics
├── +viz/             # plots, confusion matrices, error-budget curves
└── ErrorBudgetApp.mlapp
```

### Handoff contract

Python writes a single HDF5 (or `.mat` v7.3) file containing:
- trained parameters (phase mask arrays, or mesh phase angles)
- geometry metadata: grid size, physical extent, layer separations
- wavelength and any other operating-point constants
- the frozen test set and its labels
- a schema version string

MATLAB reads this file and never writes back into the Python pipeline.

**This boundary is one-directional by design.** It enforces the separation between the
design model (Python, ideal) and the as-built model (MATLAB, imperfect). Do not add a
reverse path. Do not implement training in MATLAB. Do not implement error modeling in
Python.

---

## Objectives by phase

### Phase 1 — Wave optics foundation
- Angular spectrum method implemented and verified against analytic results
- Sampling criterion understood and enforced programmatically
- Fresnel and Fraunhofer implemented as approximations of angular spectrum, with validity ranges made explicit
- Field/unit bookkeeping established that the rest of the project rests on

Deliverable: interactive diffraction explorer (Plotly, website-embeddable). Controls for
aperture, distance, wavelength, grid size. Flags sampling violations live.

### Phase 2 — Diffractive network, ideal case
- Phase 1 propagator recast as a differentiable layer
- Stack of phase masks trained to classify
- Input encoding scheme chosen deliberately (amplitude, phase, or both)
- Detector regions with integrated intensity readout
- Optical power budget established: photons per detector region per inference

Deliverable: trained D²NN, plus a written physical interpretation of what the masks do
optically, plus the power budget, plus an explicit statement of the expressivity limit
imposed by linearity.

### Phase 3 — MZI mesh
- MZI transfer matrix derived from coupler and phase-shifter primitives, unitarity verified
- Clements decomposition implemented and verified by reconstruction
- SVD layer (U·Σ·V†) for arbitrary real matrices
- Small mesh network trained on a toy task
- Direct comparison against the D²NN: parameter count, depth, footprint, failure modes

Optional branch: single-photon input through the same mesh (boson sampling). Same transfer
matrix, different input state statistics.

Deliverable: mesh programming toolkit — decomposition, verification, topology
visualization with per-MZI phase settings rendered.

### Phase 4 — Error budget
- Each hardware imperfection modeled independently, then jointly
- Every error magnitude traced to a published measurement, cited inline in code
- Monte Carlo over realizations, accuracy statistics collected
- Tolerance curves produced

Error sources, in implementation order:
1. Phase shifter error (Gaussian σ per setting)
2. Quantization (6/8/10/12-bit DAC resolution)
3. Coupler imbalance (deviation from 50:50)
4. Loss (insertion loss per MZI, propagation loss per cm)
5. Wavelength drift and dispersion
6. Thermal crosstalk (distance-dependent coupling matrix)
7. Detector noise (shot noise from Phase 2 photon budget, thermal noise, ADC quantization)

Deliverables: App Designer dashboard with per-source sliders, live accuracy, confusion
matrix, and a spatial sensitivity map. Plus a tolerance document stating required
precision per component to hold accuracy above a threshold.

---

## Scope boundaries

### Build

- Scalar diffraction theory only
- Idealized component models parameterized by literature-sourced values
- One classification task, kept simple (MNIST or smaller). Reuse it across all phases so results are comparable
- Analytic validation tests for every physics function
- Sampling and unitarity checks as runtime assertions, not just tests
- Citations as inline comments next to every physical constant

### Do not build

- Full-wave electromagnetic simulation (FDTD, FEM). If real component S-parameters are ever wanted, they get imported as data — the solver is not part of this project.
- Vector/polarization-resolved propagation. Scalar only.
- Nonlinear optical materials or physical activation functions. The nonlinearity limitation is to be characterized and documented, not engineered around.
- Convolutional or otherwise elaborate electronic network layers. The electronic side stays at "integrate intensity, softmax." If the model needs a bigger electronic head to work, that is a finding, not a problem to fix.
- Multiple datasets or a benchmarking suite. One task.
- In-situ / hardware-in-the-loop training. In-silico training then transfer is the entire premise.
- Training or optimization in MATLAB.
- Error modeling in Python.
- Any invented numerical value for a physical parameter. If a value cannot be sourced, mark it `# UNSOURCED` and surface it rather than burying it.
- Layout, mask files, foundry submission artifacts. Nothing is being fabricated.

### Deliberately deferred

These may be revisited only after Phase 4 is complete:
- Lensless imaging / phase retrieval branch (reuses `propagate.py`)
- Boson sampling depth beyond the basic Phase 3 branch
- Reservoir computing variant
- Importing measured S-parameters for a single real coupler

---

## Working conventions

- Physics functions are pure and NumPy-native; the torch/jax wrappers in `layers.py` are thin. Do not entangle autodiff machinery with the physics modules.
- Every function in `propagate.py` and `mzi.py` has a corresponding analytic test in `validate.py`.
- Field objects carry their physical units. No bare arrays crossing module boundaries.
- Prefer explicit, readable physics over vectorized cleverness. This codebase is a portfolio artifact and is meant to be read.
- Random seeds fixed and recorded for every Monte Carlo run.

---

## Open decisions

Still open. Do not assume an answer; ask.

3. **Quantum branch placement.** Inline in Phase 3, or a separate deeper Phase 5.

### Resolved

Kept here so a later session does not reopen a question the project already answered.

1. **PyTorch or JAX** → **PyTorch.** Used throughout: `photonn/layers.py`, `models.py`,
   `train.py`. The error model lives in MATLAB and is never differentiated through, so JAX's
   composability advantage never came due.
2. **Phase 4 ordering** → **error budget first.** Run against the trained D²NN before Phase 3
   was complete; see `docs/tolerance_d2nn.md`. The MZI-specific sources (coupler imbalance,
   per-MZI loss) stay deferred stubs, since they have no meaning for phase masks.
4. **Parameter source standardization** → **resolved for the D²NN**, still open for the mesh.
   The canonical set is the SLM / phase-plate and sCMOS measurement literature, not an
   integrated-photonics PDK; the ledger is `docs/parameter_sources.md`. The Phase-3 mesh will
   need its own PDK-anchored set.
