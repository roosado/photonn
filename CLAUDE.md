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
├── mzi.py            # 2x2 MZI transfer matrix, Clements/Reck decomposition, mesh forward pass,
│                     # and the Phase-5 electro-optic activation (an MZI driven by its own light)
├── layers.py         # differentiable (torch/jax) wrappers around propagate/elements/mzi
├── models.py         # D2NN and mesh network definitions
├── train.py          # training loops, datasets, input encoding schemes
├── detect.py         # detector regions, photon budget, shot and thermal noise
├── export.py         # serialize trained parameters for MATLAB handoff
├── handoff.py        # read one back: the single Python reader of that file
└── validate.py       # analytic test cases and invariant checks

photonn-hw/
├── +io/              # load exported parameter sets and frozen test data
├── +err/             # device error (phase, quantization, dispersion, loss, crosstalk)
│                  # and geometry error (spacing, registration, calibration, detector)
├── +mc/              # Monte Carlo drivers and statistics
├── +viz/             # plots, confusion matrices, error-budget curves
└── ErrorBudgetApp.mlapp
```

### Handoff contract

Python writes a single HDF5 (or `.mat` v7.3) file containing:
- trained parameters (phase mask arrays, or mesh phase angles)
- geometry metadata: grid size, physical extent, layer separations, and — since
  schema 0.3.0 — the detector layout, so the as-built model reads where the
  detectors sit rather than re-deriving it from constants typed on both sides
- the operating point: a **closed set**, listed in `photonn.export.OPERATING_POINT`
  with the model kinds that require each constant. The writer rejects an
  unrecognised key and a missing required one; both readers refuse a file that
  lacks one rather than defaulting it. A default here is indistinguishable from a
  correct value downstream, which is how a renamed field used to become a silent
  10× rescale instead of an error
- the frozen test set and its labels
- the trained accuracy, and a schema version string

MATLAB reads this file and never writes back into the Python pipeline. Python
reads it too, through `photonn/handoff.py` — that is not a reverse path, and it
replaced four modules each opening h5py and re-stating the schema by hand.

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

Those seven are **device** errors. The D²NN also has **geometry** errors, added after the
device half was complete — where the parts sit rather than what is wrong inside them. They
are a separate family because they are a different problem for a builder: a device error is
fixed once the part is made, an alignment error is set at assembly and can sometimes be
calibrated out. The mesh has no equivalent; lithography places its waveguides.

8. Plane spacing (independent per-gap deviation from the nominal z)
9. Lateral mask registration (sub-pixel displacement per plate)
10. Systematic phase gain (a calibration error, not a setting error)
11. Detector lateral offset (axial offset is the last gap, already in 8)

Deliverables: App Designer dashboard with per-source sliders, live accuracy, confusion
matrix, and a spatial sensitivity map. Plus a tolerance document stating required
precision per component to hold accuracy above a threshold.

### Phase 5 — Electro-optic activation (mesh only)

Phases 2–4 measured the linearity wall: stacked optics collapse to one linear map, so two
meshes with nothing between them do no better than one. Phase 5 puts one physical
nonlinearity between mesh layers and asks the project's question of the result.

- The device is the electro-optic activation of Williamson, Hughes, Minkov, Bartlett, Pai
  & Fan, *IEEE JSTQE* 26(1):7700412 (2020), doi:10.1109/JSTQE.2019.2930455. A tap coupler
  sends a fraction α of each mode's light to a photodiode; the amplified photocurrent sets
  the internal phase of an MZI that the rest of the same light crosses. The signal stays
  optical and coherent: no fresh beam, no digital step between layers
- It is our own MZI with a power-driven phase, and is verified against the `mzi`
  primitives the way Clements is verified by reconstruction (agreement to 1.8e-15 was
  measured in a scratch probe, 2026-10-04)
- **Gate first.** Reproduce the paper's linear-versus-activated gain inside this framework
  before anything enters the library. The first probe (36 modes, 5 epochs) did *not*
  reproduce it. If the gain does not reproduce, that is the phase's finding: write it up
  and stop
- Then the Phase-4 question, asked of the new machine: how precisely must the activation
  be built, and what does crossing the wall cost the mesh's existing tolerances. Absolute
  optical power stops cancelling in the readout once an intensity-dependent element sits
  mid-network, so loss and the photon budget are re-measured, not carried over
- Mesh only. A photodiode-and-modulator loop per pixel between D²NN plates is not a
  free-space device anyone builds; the D²NN's linearity limit stays as documented
- The task stays MNIST and the readout stays "integrate intensity, softmax". The only
  model-side change is a second mesh layer

Deliverable: a two-layer activated mesh with an as-built correctness anchor, the
activation's own tolerance edges, the mesh edges re-measured at the new depth, and a site
page presenting both — or, if the gate fails, a written null result.

---

## Scope boundaries

### Build

- Scalar diffraction theory only
- Idealized component models parameterized by literature-sourced values
- One classification task, kept simple (MNIST or smaller). Reuse it across all phases so results are comparable
- Analytic validation tests for every physics function
- Sampling and unitarity checks as runtime assertions, not just tests. Enforced:
  the propagators call `validate.assert_sampling` and `clements_decompose` checks
  its own result. A caller deliberately outside the criterion says so with
  `validate.relaxed()` rather than the check simply not existing
- Citations as inline comments next to every physical constant

### Do not build

- Full-wave electromagnetic simulation (FDTD, FEM). If real component S-parameters are ever wanted, they get imported as data — the solver is not part of this project.
- Vector/polarization-resolved propagation. Scalar only.
- Nonlinear optical materials (Kerr media, saturable absorbers, phase-change cells), and any physical activation other than Phase 5's electro-optic one. The linearity limitation is still characterised and documented, not engineered around: Phase 5 admits one activation as a *device to put a tolerance on*, not as an accuracy fix, and any accuracy it buys is reported next to what it costs. The D²NN gets no activation.
- Convolutional or otherwise elaborate electronic network layers. The electronic side stays at "integrate intensity, softmax." If the model needs a bigger electronic head to work, that is a finding, not a problem to fix. The Phase-5 activation's photodiode and amplifier are analog, per-mode and weightless — part of the device, not a layer. Nothing digital sits between optical layers.
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

3. **Quantum branch placement.** Inline in Phase 3, or a separate deeper phase after
   Phase 5 (which is now the electro-optic activation).

### Resolved

Kept here so a later session does not reopen a question the project already answered.

1. **PyTorch or JAX** → **PyTorch.** Used throughout: `photonn/layers.py`, `models.py`,
   `train.py`. The error model lives in MATLAB and is never differentiated through, so JAX's
   composability advantage never came due.
2. **Phase 4 ordering** → **error budget first.** Run against the trained D²NN before Phase 3
   was complete; see `docs/tolerance_d2nn.md`. **The mesh budget has since been run too**
   (`docs/tolerance_mesh.md`): the MZI-specific sources are no longer stubs — coupler
   imbalance is implemented and binds, and per-MZI loss became `err.mzi_loss`, which is a
   different source from the D²NN's rather than the same one renamed. Headline: the mesh
   needs its phases **10× more accurate** (0.03 rad against 0.3), and a quarter of its U
   mesh sits outside the readout's light cone and needs no tolerance at all.
   **The D²NN's geometry half has since been run too** (2026-08-17): plate registration at
   0.10 px is the tightest number in the study, three sources at their own edges **fail
   together**, and the 33 µm connectivity bound turned out not to be a tolerance. Phase 4 is
   complete for the D²NN; the mesh needs no geometry sources.
4. **Parameter source standardization** → **resolved for the D²NN**, still open for the mesh.
   The canonical set is the SLM / phase-plate and sCMOS measurement literature, not an
   integrated-photonics PDK; the ledger is `docs/parameter_sources.md`. The Phase-3 mesh will
   need its own PDK-anchored set. **The mesh budget ran ahead of that sourcing on purpose**:
   it publishes measured tolerance *edges* (properties of the network and its topology, which
   will not move) with every realistic as-built value marked `UNSOURCED` and no margin column.
   The gap is a table in `docs/parameter_sources.md`, not a silence.
5. **Physical nonlinearity** → **admitted once, on the mesh** (2026-10-04). Williamson et
   al.'s electro-optic activation (Phase 5), chosen because it is built from parts the mesh
   already has and its parameters — responsivity, amplifier gain, V_π — are electronics
   with datasheets rather than material constants. Materials and the D²NN stay out.
   In-situ training stays out too, which is why the on-chip version (Bandyopadhyay et al.,
   *Nat. Photon.* 18:1335 (2024), trained in situ) is evidence that the device exists, not
   a method to copy.

---

## Agent skills

### Issue tracker

GitHub Issues on `roosado/photonn`. Reads run directly; writes are handed to the
user as a script rather than executed. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles, each label string equal to its name. See
`docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root, both created
lazily. See `docs/agents/domain.md`.
