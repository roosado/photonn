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

## Open decision — RESOLVED for the D²NN

**#4 — canonical parameter source.** For the diffractive (D²NN) error budget the
device is a **visible-wavelength phase-mask processor** (532 nm, ~8 µm pitch),
physically realised as a liquid-crystal-on-silicon (LCoS) spatial light modulator
or a lithographic phase plate, read out on a scientific-CMOS detector. The
canonical sources are therefore the **SLM / phase-plate and sCMOS measurement
literature and datasheets** below — not an integrated-photonics MZI PDK. The MZI
mesh (Phase 3) will need its own PDK-anchored set; that remains open.

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
- No invented numeric constants are committed in code: `+err` functions take
  magnitudes as arguments; the values above enter only via the driver sweeps and
  this ledger.
