# ErrorBudgetApp (to be authored in App Designer)

The Phase-4 deliverable dashboard. **A `.mlapp` is a binary App Designer file and
cannot be scaffolded as text** — it must be created interactively in MATLAB. This
note is the placeholder and spec; `launch_error_budget.m` is the scripted entry
point that will start it.

## Create it

1. In MATLAB: **App Designer** → new Blank App → save as `ErrorBudgetApp.mlapp`
   in this folder (`photonn-hw/`).
2. Build the layout below, wiring callbacks to the `+err`, `+mc`, and `+viz`
   package functions.
3. Launch via `launch_error_budget` (adds the folder to the path and runs the app).

## Planned layout (from CLAUDE.md Phase 4)

- **Load** a handoff file via `io.read_handoff`.
- **Per-source sliders** — one control per `+err` source (phase-shifter sigma,
  DAC bits, coupler epsilon, loss, wavelength drift, thermal crosstalk, detector
  noise).
- **Live accuracy** readout on the frozen test set as sliders move.
- **Confusion matrix** (`viz.confusion_matrix`).
- **Spatial sensitivity map** (`viz.sensitivity_map`).
- Backed by `mc.run_montecarlo` for the statistics.

## Companion output

A tolerance document stating the precision required per component to hold accuracy
above a chosen threshold (built from `viz.tolerance_curve`).
