# Wave optics foundation (Phase 1)

The scalar-diffraction core in `photonn/propagate.py`, its analytic references in
`photonn/validate.py`, and the sampling criteria that bound their validity. This
document is the "made explicit" artifact for Phase 1: what each propagator
computes, where it is valid, and how sampling is enforced.

Scalar theory only — no vector/polarization effects (project scope boundary).
All quantities are SI; lengths in metres. Fields are centre-origin
(`Field.coords()` puts the origin at index `n//2`); every propagator uses
`fftshift`/`ifftshift` so input and output planes stay centred.

## Angular spectrum method (reference)

A field `U0(x, y)` is decomposed into plane waves `A(fx, fy) = FT{U0}`, each
advanced by its axial phase, then recombined:

```
U(x, y; z) = IFT{ FT{U0} · H(fx, fy; z) },
H = exp( i · 2π · z · sqrt(1/λ² − fx² − fy²) ).
```

- **Propagating vs evanescent.** Where `fx² + fy² > 1/λ²` the square root is
  imaginary; computing it as a complex sqrt makes those components *decay* with
  `z` (evanescent), as physics requires.
- **No paraxial approximation.** ASM is exact up to the grid band limit, so it is
  the reference against which Fresnel, Fraunhofer, and the Phase-2 torch layer are
  checked.
- **Band limiting.** For large `z` the transfer function `H` oscillates faster
  than the frequency grid can sample, causing aliasing. We zero `H` beyond the
  local-frequency limit

  ```
  u_limit = 1 / ( λ · sqrt( (2 · Δu · z)² + 1 ) ),   Δu = 1/(n·dx),
  ```

  applied separably in `fx` and `fy` (Matsushima & Shimobaba 2009). This keeps the
  method alias-free into the far field, at the cost of discarding content beyond
  the propagating band.

Reference: Goodman, *Introduction to Fourier Optics* (3rd ed.), §3.10; Matsushima
& Shimobaba, IEEE TIP 18(11):2646 (2009).

## Fresnel (paraxial near field)

Transfer-function form on the **same grid** as ASM:

```
H_Fresnel = exp(i k z) · exp( −i π λ z (fx² + fy²) ),
```

the paraxial (small-angle) expansion of the ASM transfer function. Because it
shares the grid, it is directly comparable to ASM pixel-for-pixel — which is how
`test_fresnel_agrees_with_angular_spectrum_in_validity_range` validates it.

**Valid when:** angles are small (paraxial) *and* the transfer function is well
sampled (`|z| ≤ z_crit`, below). For the beams here the two agree to ~1e-8.

## Fraunhofer (far field)

Single-FFT far-field form. The observation plane is **resampled** to

```
dx' = λ z / (n · dx),
```

and the field is the (prefactor-weighted) Fourier transform of the input. A
circular aperture of radius `a` produces the Airy pattern

```
U(r') ∝ 2 J₁(x)/x,   x = k a r'/z,
```

with the first null at `r' = 1.22 · λ z / (2a)`.
`test_fraunhofer_circular_aperture_is_airy` checks this against
`validate.airy_pattern` on the shared output grid.

**Valid when:** `|z| ≥ z_Fraunhofer = 2 D² / λ`, with `D` the source extent.

Reference: Goodman §4.2 (Fresnel), §4.3–4.4 (Fraunhofer, Airy).

## Sampling criteria (`check_sampling`)

Enforced programmatically and surfaced live in the diffraction explorer.

| Method | Well-sampled when | Meaning |
|--------|-------------------|---------|
| `angular_spectrum`, `fresnel` | `\|z\| ≤ z_crit = n·dx²/λ` | transfer function adequately sampled |
| `fraunhofer` | `\|z\| ≥ 2·D²/λ` (`D = n·dx`) | far enough for the far-field approximation |

`z_crit` is the critical distance where the frequency-domain (transfer-function)
approach and the space-domain approach exchange adequacy (Voelz, *Computational
Fourier Optics*, ch. 5). Note `z_crit` depends on `λ`, `dx`, `n` but **not** on the
aperture — which is why the explorer's aperture control does not move the
sampling threshold.

`check_sampling` returns a `SamplingReport(ok, method, messages)`;
`validate.assert_sampling` raises when `not ok`. Beyond `z_crit`, ASM does not
fail loudly — the band limit above absorbs the aliasing — so the explorer *flags*
the crossing rather than the code refusing to run.

## Numeric verification (observed)

From the Phase-1 test suite and spot-checks (633 nm, 128²–1024² grids):

- ASM vs analytic Gaussian at `z = 0.1 m`: normalized-intensity RMS error ≈ 2e-8;
  power conserved to 1e-6; `w(z)` matches analytic to 5 significant figures.
- Fresnel vs ASM in the paraxial, well-sampled range: max normalized-intensity
  difference ≈ 9e-9.
- Fraunhofer of a circular aperture vs Airy: max difference ≈ 1e-3 over the bright
  central region; first null on the predicted ring.

## What Phase 1 does not cover

Trainable phase masks and the differentiable torch propagator (Phase 2), detector
regions / photon budget (Phase 2), MZI meshes (Phase 3), and any hardware error
model (Phase 4). `elements.phase_mask` / `amplitude_mask` remain Phase-2 stubs.
