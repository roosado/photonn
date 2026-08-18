# Parameter sources

Every physical constant and error magnitude used anywhere in the project must
trace to a published measurement, cited **inline** next to where it is used. This
file is the consolidated ledger.

## Convention

- In code, put the citation on the line the constant is defined:
  - Python: `phase_sigma = 0.01  # rad, 1-sigma. Author et al., Journal Year, doi:...`
  - MATLAB: `phaseSigma = 0.01;  % rad, 1-sigma. Author et al., Journal Year, doi:...`
- If a value is needed but cannot yet be sourced, **do not invent it**. Mark it
  and surface it, do not bury it:
  - Python: `# UNSOURCED`
  - MATLAB: `% UNSOURCED`
- Add a row here for every sourced value so magnitudes stay internally
  consistent across the Python and MATLAB sides.

## Open decision — RESOLVED for the D²NN, OPEN for the mesh

**#4 — canonical parameter source.** For the diffractive (D²NN) error budget the
device is a **visible-wavelength phase-mask processor** (532 nm, ~8 µm pitch),
physically realised as a liquid-crystal-on-silicon (LCoS) spatial light modulator
or a lithographic phase plate, read out on a scientific-CMOS detector. The
canonical sources are therefore the **SLM / phase-plate and sCMOS measurement
literature and datasheets** below — not an integrated-photonics MZI PDK. The MZI
mesh (Phase 3) will need its own PDK-anchored set; that remains open.

**The mesh budget was run anyway, and deliberately so.** `docs/tolerance_mesh.md`
reports measured tolerance **edges** for all seven mesh sources with the realistic
as-built column left `UNSOURCED` throughout, and no "Margin" column at all. The
reasoning: an edge is a property of the trained network and its topology, so it can
be measured now and does not change when the sourcing lands; a margin is a claim
about a fabrication process, and there is nothing to base one on yet. Ordering the
work this way surfaces the gap instead of stalling on it. **What is still missing is
the whole right-hand column of that table** — see the mesh ledger below.

## Ledger

Ranges are what the literature reports; "representative" is the single value used
when a nominal as-built operating point is needed. Sweep *ranges* in
`run_error_budget.m` are deliberately wider than the realistic values so the
tolerance edge is visible.

| Quantity | Symbol / units | Value (representative; range) | Source | Used in |
|---|---|---|---|---|
| Per-pixel phase-setting error | σ_φ (rad, 1σ) | 0.05; ~0.06–0.63 (λ/100–λ/10 RMS), flicker ~0.003 (0.001π) | LCoS phase-modulation characterisation, *Appl. Sci.* 9(13):2592 (2019); Laser2000 LCoS spec guide | `err.phase_shifter_error`; phase sweep in `run_error_budget.m` |
| DAC / addressing resolution | n_bits | 8; 8–16 (0–255 … 0–4095 grey levels) | Laser2000 LCoS SLM specification guide | `err.quantize` |
| Laser wavelength drift | Δλ (nm) | <0.1; ≈2e-5 (TEC-locked, 19.7 MHz) up to ~0.1–0.3 nm/°C (free-running diode) | TEC wavelength-locking, *Sensors* PMC10255239; Thorlabs 532 nm DPSS | `err.wavelength_dispersion` |
| Detector read noise | σ_read (e⁻) | 2; 1–2 (≈1 with CMS) | Teledyne sCMOS; Andor Zyla/Neo sCMOS brochure | `err.detector_noise` |
| Detector full well | N_fw (e⁻) | — (auto-ranged ADC); 2×10⁴–1.2×10⁵ | Teledyne "Bit Depth, Full Well, Dynamic Range" | `err.detector_noise` (ADC full-scale) |
| ADC resolution | bits | 12; 11–16 | Teledyne sCMOS; Andor sCMOS brochure | `err.detector_noise` |
| Photon energy | E_ph = hc/λ (J) | 3.73×10⁻¹⁹ at 532 nm (exact SI h, c) | SI 2019 defining constants | `photonn/detect.py`, `model.evaluate`, `err.detector_noise` |
| Insertion loss | (dB / element) | ~1; <0.2 (reflectivity-enhanced LCoS) to ~1 | LCoS WSS study, *Photonics* 4(2):22 (2017); Laser2000 | `err.loss` |
| Propagation loss (air) | (dB/cm) | ≈0 | free-space; negligible over cm paths | `err.loss` |
| Pixel / thermal crosstalk | length scale (px) | ~1 (fringing-field blur of sharp phase edges) | Fringing-field crosstalk, *J. Eur. Opt. Soc.-RP* (2021), doi:10.1186/s41476-021-00174-7; *Appl. Opt.* 52(28):6877 (2013) | `err.thermal_crosstalk` |

Sources (URLs):
- LCoS phase modulation, *Appl. Sci.* 9(13):2592 — https://www.mdpi.com/2076-3417/9/13/2592
- Laser2000 LCoS SLM specification guide — https://photonics.laser2000.co.uk/blogs/understanding-the-jargon-of-lcos-spatial-light-modulators-slms-blog/
- TEC laser wavelength locking, *Sensors* — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10255239/
- Thorlabs 532 nm DPSS lasers — https://www.thorlabs.com/NewGroupPage9_PF.cfm?ObjectGroup_ID=5597
- Teledyne — Bit depth, full well, dynamic range — https://www.teledynevisionsolutions.com/learn/learning-center/imaging-fundamentals/bit-depth-full-well-and-dynamic-range/
- Andor Zyla/Neo sCMOS brochure — https://andor.oxinst.com/downloads/uploads/Andor_sCMOS_Brochure.pdf
- LCoS WSS insertion loss, *Photonics* 4(2):22 — https://www.mdpi.com/2304-6732/4/2/22
- Fringing-field pixel crosstalk, *J. Eur. Opt. Soc.-RP* — https://link.springer.com/article/10.1186/s41476-021-00174-7
- Fringing-field diffraction efficiency, *Appl. Opt.* 52(28):6877 — https://opg.optica.org/ao/abstract.cfm?URI=ao-52-28-6877

## External / motivational

These do not parameterise anything the code computes — they are the **outside
context** the site's "Why build a computer out of light?" section stands on. They
live in this ledger rather than only in the page so that the same rule applies to
them as to everything else: verified before use, and dropped rather than
approximated if they cannot be confirmed. Verified 2026-08-10.

| Quantity | Value | Source | Used in |
|---|---|---|---|
| Energy per 32-bit add, 45 nm / 0.9 V | 0.9 pJ floating-point; ≈1/9 of that fixed-point | Horowitz, ISSCC 2014 | `site/index.html` — "why light" |
| Energy per 32-bit SRAM read, 45 nm | 5 pJ | Horowitz, ISSCC 2014 | as above |
| Energy per 32-bit DRAM read, 45 nm | 640 pJ | Horowitz, ISSCC 2014 | as above |

Sources (URLs):
- M. Horowitz, "Computing's energy problem (and what we can do about it)," *ISSCC*
  2014, 10–14 — https://www.semanticscholar.org/paper/947620a1854655ed91a86b90d12695e05be85983
- X. Lin et al., "All-optical machine learning using diffractive deep neural
  networks," *Science* **361**, 1004 (2018), doi:10.1126/science.aat8084 —
  https://www.science.org/doi/10.1126/science.aat8084
- Y. Shen et al., "Deep learning with coherent nanophotonic circuits," *Nature
  Photonics* **11**, 441 (2017), doi:10.1038/nphoton.2017.93 —
  https://www.nature.com/articles/nphoton.2017.93
- G. Wetzstein et al., "Inference in artificial intelligence with deep optics and
  photonics," *Nature* **588**, 39 (2020), doi:10.1038/s41586-020-2973-6 —
  https://www.nature.com/articles/s41586-020-2973-6
- W. R. Clements et al., "Optimal design for universal multiport interferometers,"
  *Optica* **3**, 1460 (2016), doi:10.1364/OPTICA.3.001460 —
  https://opg.optica.org/optica/abstract.cfm?uri=optica-3-12-1460
- M. Reck, A. Zeilinger, H. J. Bernstein & P. Bertani, "Experimental realization of
  any discrete unitary operator," *Phys. Rev. Lett.* **73**, 58 (1994). *(Cited by
  volume/page only; no URL verified.)*
- C. C. Wanjura & F. Marquardt, "Fully nonlinear neuromorphic computing with linear
  wave scattering," *Nature Physics* (2024), doi:10.1038/s41567-024-02534-9 —
  https://www.nature.com/articles/s41567-024-02534-9 · preprint
  https://arxiv.org/abs/2308.16181. Verified 2026-08-14. *(Volume and page not
  confirmed: nature.com sits behind an auth redirect, so the page cites it by DOI.)*

### Nonlinearity routes — mostly UNSOURCED, and the page says so

`/optics` closes by naming three ways the field is trying to give an optical network
the nonlinearity a mask stack cannot have. **Only one of the three is sourced.**

| Route | Status |
|---|---|
| Input encoded in the scattering parameters rather than the wave | **Sourced** — Wanjura & Marquardt, above |
| Optoelectronic conversion loops | `UNSOURCED` — described mechanically, no citation |
| Intensity-dependent materials (saturable absorbers, phase-change) | `UNSOURCED` — described mechanically, no citation |

This is the one place on the site where a claim runs ahead of its ledger, so it is
handled the way the convention above requires: the two unsourced routes are stated
as mechanisms rather than as results, no performance number is attached to either,
and the page says in its own words that this study tested none of them and vouches
for none of them. **Anything more than that needs a verified source first.** This is
the largest open citation task in the project.

**Two derived numbers on that page, for the record.** The 60 ps transit is the
18 mm stack length divided by `c`. The ~1 fJ per inference is the 1 pW × 1 ms
shot-noise knee from [`tolerance_d2nn.md`](tolerance_d2nn.md), **not** the nominal
1 mW × 1 ms operating point — which is 1 µJ per inference and *worse* than a GPU.
The page states that explicitly so the weaker number cannot be mistaken for the
claim. Neither figure includes the laser, modulator, detectors or converters;
nothing in this project models those.

## Mesh ledger — every row `UNSOURCED`

The seven quantities the MZI-mesh error budget sweeps, what each one drives, and the
class of source that would settle it. Nothing here has a value yet; the middle column
is what the sweep covers, not a claim about any process. Filling this table in is the
**second-largest open citation task in the project**, after the nonlinearity routes.

The device the mesh budget implies is a **telecom-wavelength silicon photonic
processor** (1550 nm, thermo-optic phase shifters, directional couplers or MMIs), so
the sources will be foundry PDK data and integrated-photonics characterisation
papers — a different literature from the SLM/sCMOS set above, which is the whole
reason open decision #4 is still open for the mesh.

| Quantity | Symbol / units | Sweep range covered | What would settle it | Used in |
|---|---|---|---|---|
| Phase-shifter setting error | σ_φ, rad | 0 … 0.12 | thermo-optic shifter calibration accuracy | `err.phase_shifter_error` |
| Coupler split deviation | ε, power fraction, 1-σ | 0 … 0.12 | MMI / directional-coupler process spread across a wafer | `err.coupler_imbalance` |
| Heater thermal coupling | α, dimensionless | 0 … 0.02 | measured crosstalk between adjacent thermo-optic shifters | `err.mesh_coupling_matrix` |
| Thermal decay length | µm | **fixed at 50** | in-plane heat spreading in the device layer | `err.mesh_coupling_matrix` |
| MZI layout pitch | µm, [column, mode] | **fixed at [80, 40]** | a real mesh floorplan | `meshmodel.schedule` → `err.mesh_coupling_matrix` |
| Per-MZI insertion loss | dB per MZI | 0 … 2.0 | measured MZI insertion loss | `err.mzi_loss` |
| Waveguide propagation loss | dB/cm | not swept (0) | SOI or SiN propagation loss | `err.mzi_loss` |
| Coupler dispersion | Δsplit per nm | **fixed at 0.002** | coupler split vs wavelength characterisation | `err.mesh_wavelength_dispersion` |

The three **fixed** rows are the ones to worry about most, because they do not appear
on any curve's x-axis: the crosstalk sweep moves α with the geometry held constant, so
the whole curve slides if the pitch or the decay length is wrong. The tolerance
document says so in its Caveats rather than presenting the crosstalk edge as
geometry-independent.

DAC resolution and the detector parameters are **not** in this table. They are
electronics, already sourced in the ledger above, and they carry over to the mesh
unchanged — which is why the mesh and the D²NN measure the same 1 pW detector edge.

## Alignment ledger — every row `UNSOURCED`

The four quantities the D²NN's **geometry** sweeps cover (added 2026-08-17,
issue #6), what each one drives, and the class of source that would settle it.
Nothing here has a value yet, for the same reason the mesh table below has none: the
study publishes measured *edges*, which are properties of this network and will not
move, and refuses to invent the other half of a margin.

The literature that would settle these is **optomechanical**, and it is a third
distinct set from the SLM/sCMOS values in the ledger above and the integrated-photonics
PDK data the mesh needs: kinematic-mount repeatability, translation-stage resolution
and drift, opto-mechanical stability over a temperature cycle. None of it is sourced.

| Quantity | Symbol / units | Sweep range covered | What would settle it | Used in |
|---|---|---|---|---|
| Lateral plate registration | σ, pixels (8 µm each) | 0 … 1.0 px | mount repeatability and assembly registration for a stacked plate | `err.mask_registration` |
| Detector lateral placement | σ, pixels | 0 … 6 px | sensor-mount placement accuracy against the optical axis | `err.detector_offset` |
| Plane spacing, per gap | σ, µm | 0 … 300 µm | spacer tolerance or stage resolution along the axis | `err.plane_spacing` |
| Phase calibration gain | k, dimensionless | ±0 … ±75 % | SLM phase-response calibration accuracy over its stroke | `err.phase_gain` |

**Registration is the one that matters.** Its measured edge, 0.10 px = **0.8 µm per
plate**, is the tightest requirement anywhere in the study, and whether it is
achievable is precisely what these missing sources would decide. The other three
have edges (16 µm, 100 µm, ±10 %) that are loose enough to be plausibly comfortable,
but "plausibly" is not a margin and the table does not pretend otherwise.

**Plate tilt and rotation are not in the table because they are not modelled.**
Registration covers translation only. A rotated plate induces a displacement that
grows with distance from the axis, so it is a different error and probably a worse
one; it is named here so its absence is on the record rather than implied.

## Outstanding `UNSOURCED` / modelling choices

Not every number is a directly-measured constant; some are deliberate modelling
choices flagged here so nothing is passed off as measured:

- **Thermal-crosstalk kernel shape.** The fringing-field crosstalk is real and
  cited above, but modelling it as a **Gaussian blur of a given σ (pixels)** is a
  simplification; the σ-to-physical-crosstalk mapping is approximate. The sweep
  reports accuracy vs blur σ rather than claiming a single measured σ.
- **Per-pixel phase σ.** The λ/100–λ/10 RMS figures are *aperture-averaged*
  wavefront RMS; treating them as i.i.d. per-pixel σ is a conservative modelling
  choice (real errors are spatially correlated).
- **Mesh heater coupling kernel.** `err.mesh_coupling_matrix` uses
  `C_ij = α·exp(-d_ij / L)`, the standard lumped form for in-plane heat spreading in
  a thin device layer — not a solution of the heat equation for this geometry. The
  decay length is doing all the work and is not sourced.
- **Mesh thermal power proxy.** Crosstalk is driven by each MZI's *programmed phase*
  as a stand-in for its dissipated heater power. That is exact for a shifter whose
  phase is proportional to applied power, which is the usual thermo-optic case, and
  wrong for one driven in voltage.
- **V and U modelled as thermally independent.** A layout assumption, and the
  optimistic one.
- **Geometry errors are drawn independently per part.** Each plate is mounted
  separately, so each is displaced separately, and each gap is set separately. That
  is the right model for an assembly built one piece at a time and the wrong one for
  a monolithic mount whose errors would be common-mode — and common-mode is the
  benign case, since a translation of the whole stack largely cancels at a detector
  in the same frame. The independent draw is therefore the conservative choice.
- **A displaced plate is still phase-only.** `err.mask_registration` translates the
  transmittance `exp(iφ)` and keeps `angle()` of the result, discarding the modulus
  that Fourier interpolation introduces. That is physically right rather than merely
  convenient: a real plate that moves does not stop being phase-only.
- **The 33 µm connectivity floor is not a tolerance.** `phase3_mesh.md` derives it as
  a bound on what *can* couple; sweeping it directly shows accuracy is unchanged at
  the floor (0.7725 against 0.7712 nominal), because the light that stops coupling is
  the Nyquist-ray corner, which carries almost no power. Recorded here because the
  number reads like a tolerance and is not one.
- No invented numeric constants are committed in code: `+err` functions take
  magnitudes as arguments; the values above enter only via the driver sweeps and
  this ledger. The three fixed mesh geometry constants live in
  `run_error_budget_mesh.m`, each marked `% UNSOURCED` on the line that sets it.
