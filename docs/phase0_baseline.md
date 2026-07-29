# Phase 0 — scaffolding baseline

Snapshot of the verified scaffolding state. This is the foundation every later
phase builds on; the numbers below were confirmed by running the suite, not
asserted. Date: 2026-07-23.

> **Historical snapshot — superseded by Phase 1.** This records the Phase-0
> scaffolding state before any physics existed. Phase 1 has since landed:
> `propagate.py`, `elements.py`, `validate.py`, and `apps/diffraction_explorer.py`
> are now implemented and tested (`24 passed, 4 skipped`; the 4 skips are Phase-3
> MZI placeholders). The "Implemented vs stubbed" and test-count figures below
> describe Phase 0 only and are kept for the record — for current state see
> [`../README.md`](../README.md) and [`wave_optics.md`](wave_optics.md).

## What Phase 0 delivered

- Full directory structure for both codebases (`photonn/` Python, `photonn-hw/`
  MATLAB), matching the architecture in `CLAUDE.md`.
- Importable Python package with fixed public API signatures for every module.
- A green test harness (`pytest`), with physics tests present but skipped.
- The one-directional HDF5 handoff contract, **implemented and tested** — the
  highest-risk interface, locked before any physics exists.

## Implemented vs stubbed

**Implemented (real, tested code):**

- `photonn/fields.py` — `Field` container: units/geometry bookkeeping
  (`n`, `extent`, `k`, `coords`, `intensity`, `phase`, `power`) with input
  validation. The unit backbone the rest of the project rests on.
- `photonn/export.py` — `write_handoff` / `validate_handoff` and
  `SCHEMA_VERSION`. Authoritative writer/validator for the contract in
  `docs/handoff_schema.md`.
- `photonn-hw/+io/read_handoff.m` — MATLAB reader with schema-version check
  (not runnable in the Python CI environment; verify in MATLAB).

**Stubbed (raise `NotImplementedError`, full docstrings + signatures):**

- `propagate.py`, `elements.py`, `mzi.py` — NumPy physics (Phase 1 / Phase 3).
- `layers.py`, `models.py`, `train.py`, `detect.py` — torch models + training
  (Phase 2 / Phase 3).
- `validate.py` — analytic references + `assert_*` invariant helpers.
- `apps/diffraction_explorer.py` — Phase-1 Plotly deliverable.
- `photonn-hw/+err`, `+mc`, `+viz` — Phase-4 error model, Monte Carlo, plots.

## Verification results

Run from `D:\Python\Photonn` in the project venv (`.venv`, Python 3.12.0):

| Check | Command | Result |
|-------|---------|--------|
| Imports (incl. torch-backed `layers`) | `python -c "import photonn, photonn.layers, photonn.export"` | OK — `0.1.0`, schema `0.1.0` |
| Test suite | `pytest -q` | **13 passed, 10 skipped** (~0.8 s) |
| Handoff round-trip | write → `validate_handoff` → reopen with `h5py` | groups/shapes/dtypes/attrs intact |
| Stubs raise | call `propagate.angular_spectrum`, `mzi.mzi_matrix` | `NotImplementedError` (wired, not silent) |

- The **13 passing** tests: 6 handoff round-trip + 7 `Field` container.
- The **10 skips**: Phase-1/3 physics placeholders, each labelled with the
  analytic invariant it will assert (energy conservation, Airy pattern,
  unitarity, Clements reconstruction, …).

## Environment

- Interpreter: `D:\Python\Photonn\.venv\Scripts\python.exe` (Python 3.12.0).
- Install: `pip install -e ".[dev]"` from `pyproject.toml`.
- Confirmed versions: numpy 2.5.1 · torch 2.13.0+cpu (CPU wheel, laptop) ·
  h5py 3.16.0 · scipy · matplotlib · plotly · pytest.

## Not verified in this environment

The MATLAB reader. When next in MATLAB:

```matlab
cd D:\Python\Photonn\photonn-hw
d = io.read_handoff('<path-to-handoff>.h5')
```

Caveat: MATLAB's `h5read` returns dataset dimensions reversed vs. NumPy
(row-major → column-major). A Python `f32[n, N, N]` comes back as `N x N x n`.

## Locked decisions

- Scope of this pass: scaffolding only.
- Autodiff library: **PyTorch**.
- Handoff `SCHEMA_VERSION = "0.1.0"` — bump on any layout change, in both
  `photonn/export.py` and the MATLAB reader's `EXPECTED_SCHEMA`.

## Next step (Phase 1, separate work)

Implement `propagate.angular_spectrum` and `check_sampling` against the analytic
references in `validate.py`, convert the corresponding skipped tests in
`tests/test_propagate.py` into real assertions, then build the
`diffraction_explorer` deliverable. See `CLAUDE.md` → "Phase 1 — Wave optics
foundation".
