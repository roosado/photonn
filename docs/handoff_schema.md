# Handoff schema (`schema_version = 0.3.0`)

The single HDF5 file Python writes and MATLAB reads. **One-directional by
design:** Python (`photonn/export.py`) writes; MATLAB (`photonn-hw/+io/read_handoff.m`)
reads and never writes back. This boundary enforces the separation between the
ideal *design* model and the imperfect *as-built* model.

The authoritative writer/validator is [`photonn/export.py`](../photonn/export.py).
Any change to this layout must bump `SCHEMA_VERSION` there **and** the
`SUPPORTED_SCHEMAS` list in the MATLAB reader.

Writers always emit the current version; readers accept every version in
`SUPPORTED_SCHEMAS`. That is what lets 0.2.0 and 0.3.0 land without re-exporting
the 131 MB `exports/d2nn_phase2.h5`.

## Layout

```
/                                 (root)
  @schema_version   str           e.g. "0.3.0" (checked by the reader)
  @created          str           ISO-8601 UTC timestamp
  @description      str           free text
  @test_acc         f64           test accuracy of the exported model (0.3.0+)
                                  Older files state it only inside @description;
                                  photonn.handoff parses that as a fallback.

/geometry
  @grid_size        int           N (field is N x N)
  @physical_extent_m float         side length of the field plane, metres
  @n_layers         int           number of parameterized planes
  layer_separations_m  f64[·]      axial gaps, metres (length convention set in Phase 2)
  detector_regions  i4[n_classes, 4]  (y0, y1, x0, x1) per class, 0-based half-open
                                  required for d2nn since 0.3.0; absent for mesh

/operating_point
  # A closed set, not an open one. Every key below is listed in
  # photonn.export.OPERATING_POINT with the model kinds that require it; the
  # writer rejects an unrecognised key and a missing required one, and both
  # readers refuse a file that lacks one rather than defaulting it.
  @wavelength_m      f64          operating wavelength, metres      (d2nn, mesh)
  @readout_gain      f64          region intensity -> logit scale   (d2nn, mesh)
  @input_power_w     f64          entrance power, photon budget     (d2nn, mesh)
  @integration_time_s f64         detector integration window       (d2nn, mesh)
  @pixel_pitch_m     f64          grid pitch, metres                (d2nn)
  @phase_scale_rad   f64          full-scale mask phase             (d2nn)
  @input_frac        f64          entrance window as a fraction of N (d2nn)
  @encoding_code     f64          0 amplitude, 1 phase, 2 both      (d2nn)
  @n_modes           f64          mesh width                        (mesh)
  @n_classes         f64          readout classes                   (mesh)
  @sigma_gain        f64          external gain undoing passivization (mesh)

/parameters
  @model_type       str           "d2nn" | "mesh"
  # model_type == "d2nn":
  phase_masks       f64[n_layers, N, N]    trained phase profiles, radians
  # model_type == "mesh"  (all four datasets required since 0.2.0):
  @n_modes          int           mesh width
  @n_mzi_per_mesh   int           n_modes(n_modes-1)/2, the Clements bound
  @mesh_order       str           "V,U" -- how the meshes are concatenated below
  @topology         str           "clements_rectangular"
  phase_theta       f64[n_meshes * n_mzi]  internal MZI phases, radians
  phase_phi         f64[n_meshes * n_mzi]  external MZI phases, radians
  sigma             f64[n_modes]           diagonal transmissions, passivized to [0, 1]
  out_phase         f64[n_meshes, n_modes] per-mesh output phase screen, radians

/test_set
  images            f32[n_samples, N, N]   frozen test images (encoded input)
  labels            i32[n_samples]         integer class labels
```

## Notes

- **Required** groups: `/geometry`, `/operating_point`, `/parameters`, `/test_set`.
  `validate_handoff` fails on the first missing group/attribute/dataset.
- **Array order.** Datasets are written row-major (C order) from NumPy. MATLAB's
  `h5read` returns them with dimensions reversed (column-major); the reader and
  any Phase-4 code must account for this — e.g. a Python `f32[n, N, N]` comes
  back as `N x N x n` in MATLAB.
- **Extensibility.** New scalar operating constants go under `/operating_point`
  as attributes without a schema bump. Structural changes (new groups, changed
  dtypes/shapes) require a version bump.
- **Test set is frozen.** The same `/test_set` is reused across phases so ideal
  and as-built accuracy are measured on identical inputs.
- **The mesh operator is `U · diag(sigma) · V`, with no conjugate transpose.** The
  prose in `docs/phase3_mesh.md` calls it `U·Σ·V†`; since V is a free unitary the
  model class is identical, but a reader must use V **as stored**. Modes run
  row-major: a `6×6` test image flattens to the 36-mode input vector in C order.
- **`sigma` is passivized on export.** The trained diagonal is signed and exceeds 1;
  `photonn.mzi.passivize` folds the sign into `out_phase[0]` and the scale into
  `/operating_point.sigma_gain`, both of which leave the logits identical. What
  crosses the boundary is therefore a device that could exist.

## Version history

### 0.3.0

* `/geometry/detector_regions` -- where the detectors sit, written as data.
  It was the one design parameter the handoff never carried: MATLAB re-derived
  it from `field_frac = 0.75` and `patch_frac = 0.11` typed into
  `+model/detector_regions.m`, and the two sides agreed only because someone
  kept two copies of the same arithmetic in step across two languages. Required
  for `d2nn`; readers fall back to deriving it for older files.
* `@test_acc` at the root, instead of a token inside the free-text description
  that two exporters parsed back out with byte-identical hand-rolled parsers.
* `/operating_point` became a closed manifest. No layout change, but the writer
  now rejects unknown and missing keys, and the MATLAB reader no longer
  substitutes a default for an absent one.


- **0.2.0** — the mesh parameter set completed: `sigma` and `out_phase` added, plus
  the four `/parameters` attributes describing width, topology and mesh order.
  0.1.0 mesh files carried the MZI angles alone, which is 108 parameters short of the
  36-mode model and cannot rebuild its operator. **Additive and mesh-only** — the
  `d2nn` layout did not move, so 0.1.0 files remain valid and load unchanged.
- **0.1.0** — initial contract: geometry, operating point, `d2nn`/`mesh`
  parameters, frozen test set.
