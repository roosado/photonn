# Handoff schema (`schema_version = 0.1.0`)

The single HDF5 file Python writes and MATLAB reads. **One-directional by
design:** Python (`photonn/export.py`) writes; MATLAB (`photonn-hw/+io/read_handoff.m`)
reads and never writes back. This boundary enforces the separation between the
ideal *design* model and the imperfect *as-built* model.

The authoritative writer/validator is [`photonn/export.py`](../photonn/export.py).
Any change to this layout must bump `SCHEMA_VERSION` there **and** the
`EXPECTED_SCHEMA` constant in the MATLAB reader.

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
  # model_type == "mesh":
  phase_theta       f64[n_mzi]             internal MZI phases, radians
  phase_phi         f64[n_mzi]             external MZI phases, radians

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

## Version history

- **0.1.0** — initial contract: geometry, operating point, `d2nn`/`mesh`
  parameters, frozen test set.
