# Handoff schema (`schema_version = 0.2.0`)

The single HDF5 file Python writes and MATLAB reads. **One-directional by
design:** Python (`photonn/export.py`) writes; MATLAB (`photonn-hw/+io/read_handoff.m`)
reads and never writes back. This boundary enforces the separation between the
ideal *design* model and the imperfect *as-built* model.

The authoritative writer/validator is [`photonn/export.py`](../photonn/export.py).
Any change to this layout must bump `SCHEMA_VERSION` there **and** the
`SUPPORTED_SCHEMAS` list in the MATLAB reader.

Writers always emit the current version; readers accept every version in
`SUPPORTED_SCHEMAS`. That is what lets 0.2.0 land without re-exporting the 131 MB
`exports/d2nn_phase2.h5`.

## Layout

```
/                                 (root)
  @schema_version   str           e.g. "0.1.0" (checked by the reader)
  @created          str           ISO-8601 UTC timestamp
  @description      str           free text

/geometry
  @grid_size        int           N (field is N x N)
  @physical_extent_m float         side length of the field plane, metres
  @n_layers         int           number of parameterized planes
  layer_separations_m  f64[·]      axial gaps, metres (length convention set in Phase 2)

/operating_point
  @wavelength_m     f64           operating wavelength, metres  (required)
  @<other>          f64           additional scalar constants may be added

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

- **0.2.0** — the mesh parameter set completed: `sigma` and `out_phase` added, plus
  the four `/parameters` attributes describing width, topology and mesh order.
  0.1.0 mesh files carried the MZI angles alone, which is 108 parameters short of the
  36-mode model and cannot rebuild its operator. **Additive and mesh-only** — the
  `d2nn` layout did not move, so 0.1.0 files remain valid and load unchanged.
- **0.1.0** — initial contract: geometry, operating point, `d2nn`/`mesh`
  parameters, frozen test set.
