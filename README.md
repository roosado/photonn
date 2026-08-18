# photonn

Physical photonic neural network simulator. A study of **fabrication tolerance**: train an
idealized optical classifier in-silico, then measure how fast it degrades as real hardware
imperfections are introduced.

> **Central question:** how precisely must a photonic processor be fabricated before it stops
> computing what it was trained to compute?

## Live

### → **[roosado.github.io/photonn](https://roosado.github.io/photonn/)**

**This neural network is made of light.** The trained diffractive network itself, running its
forward pass in your browser: draw a digit or pick one from the frozen MNIST test set, and watch
it diffract through five trained phase masks onto ten detectors. No libraries, no network,
nothing precomputed.

Four more pages read on from it:

- [**the wave optics underneath**](https://roosado.github.io/photonn/physics.html) — with a
  diffraction explorer that recomputes scalar diffraction live as you move the controls
- [**the same machine built as a chip**](https://roosado.github.io/photonn/chip.html)
- [**how precisely it would have to be built**](https://roosado.github.io/photonn/tolerance.html)
  — the study, one section per error source, each with a widget showing what that error
  physically does, split into what fabrication has to get right and where the parts have to sit,
  closing on the chip, which fails a different way
- [**how much better the optics could be**](https://roosado.github.io/photonn/optics.html) —
  what depth buys, and why more of it is not the answer

**The through-line, stated once.** A stack of phase masks is *one linear operator* no matter how
tall it is, so depth converges on the best that operator can do rather than compounding the way
depth does in an ordinary network. Eleven times the masks buys 10.5 points of accuracy and costs
2× tighter phase control, while the one fabrication error that already fails does not move at
all. The way past that ceiling is a nonlinearity, which this project characterises and
deliberately does not try to build.

Two codebases, one project, separated by a one-directional boundary:

- **`photonn/`** — Python. Pure-NumPy scalar wave-optics physics, differentiable (PyTorch)
  models, training. The *ideal design* side.
- **`photonn-hw/`** — MATLAB. Fabrication-error modeling, Monte Carlo, interactive analysis.
  The *as-built* side. Reads the handoff; never writes back.

## Timeline

The project runs in phases, and the site was re-argued each time a measurement changed what
it should be claiming. Each row links to where that work is written up in full.

| | | |
|---|---|---|
| **Jul 2026** | **Phase 0–1.** Scaffolding, then the scalar-diffraction core: angular spectrum, Fresnel, Fraunhofer, a programmatic sampling criterion, and analytic references to check them against. | [`phase0_baseline.md`](docs/phase0_baseline.md) · [`wave_optics.md`](docs/wave_optics.md) |
| **Jul 2026** | **Phase 2.** The propagator recast as a differentiable layer, stacked into a trained D²NN, plus the power budget and the linearity ceiling. | [`phase2_dnn.md`](docs/phase2_dnn.md) |
| **Jul 2026** | **Phase 3.** MZI transfer matrix, Clements and Reck decomposition, an SVD layer, and a 36-mode mesh classifier — with the argument that it and the glass stack are the same machine. | [`phase3_mesh.md`](docs/phase3_mesh.md) |
| **Jul 2026** | **Phase 4, first half.** The MATLAB as-built side reproduces the ideal accuracy exactly, then breaks it one imperfection at a time. Crosstalk binds, and it fails by 4×. | [`tolerance_d2nn.md`](docs/tolerance_d2nn.md) |
| **30 Jul** | The trained network starts running its own forward pass **in the browser**, held to PyTorch by a cross-check rather than by eye. | [`site/README.md`](site/README.md) |
| **3 Aug** | The free-space ↔ chip correspondence, and the optical stack drawn in 3D. | [`phase3_mesh.md`](docs/phase3_mesh.md) |
| **6 Aug** | **Retrained on the full MNIST set**, 0.7695 → **0.7990**. Every downstream number is re-derived from it. | [`phase2_dnn.md`](docs/phase2_dnn.md) |
| **6–8 Aug** | **The optics sweep**, which overturned the ceiling claim: 0.799 belonged to *that geometry*, not to the architecture. Depth keeps paying; a bigger grid does not. | [`phase2_dnn.md`](docs/phase2_dnn.md#what-the-optics-can-still-buy) |
| **9 Aug** | A **56-mask network at 0.9040** — measured, put in the browser beside the 5-mask one, and **not promoted**: the ten points cost 2× tighter phase control and 4.7× lower loss per mask, while the constraint that already fails does not move. | [`tolerance_d2nn.md`](docs/tolerance_d2nn.md) |
| **10 Aug** | The site becomes **five pages that lead with the machine and land on the fabrication question**, and five tolerance numbers that had been stale on the live site for months are corrected. | [`site/README.md`](site/README.md) |
| **14 Aug** | The site is **re-argued around the linearity wall**, `/tolerance` becomes one section per error source with a widget for each, and models start being named by depth rather than by status. | [`site/README.md`](site/README.md) |
| **17 Aug** | **Phase 4, second half.** The same budget run against the mesh, which fails a different way: **10× tighter on phase**, coupler imbalance binding for the first time, and a quarter of one mesh needing no tolerance at all. | [`tolerance_mesh.md`](docs/tolerance_mesh.md) |
| **17 Aug** | **The other half of Phase 4.** The budget stops being about devices only and measures where the parts *sit*: plate registration at **0.8 µm** is now the tightest number in the study, a bound quoted for three phases turns out not to be a tolerance, and three errors that each pass alone **fail together**. | [`tolerance_d2nn.md`](docs/tolerance_d2nn.md#geometry-where-the-parts-sit) |
| **17 Aug** | Every page gains an **in-page contents card**, becoming a margin rail on wide screens, and `/tolerance` splits its error sources into two families: how precisely each part is made, and where the parts sit. The second is empty and says so. | [`site/README.md`](site/README.md) |
| **17 Aug** | The detector layout **re-scored against the masks that ship**, on data that is neither trained on nor the frozen test set. The shipped boxes still win, and the 40 % of light that misses them is still not headroom — but depth turns out to make the readout almost indifferent to box size. | [`phase2_dnn.md`](docs/phase2_dnn.md#the-light-that-misses-the-boxes-is-not-headroom-re-scored-2026-08-17) |

**Where things are written down.** [`docs/`](docs/README.md) is the reference — one file per
question, indexed. [`site/README.md`](site/README.md) covers the generated pages and how the
browser ports are held to the models. [`CLAUDE.md`](CLAUDE.md) holds the architecture, the
scope boundaries and the decisions that are closed. Commit messages carry the reasoning for
individual changes.

**Still open:** promoting the 56-mask geometry; sourcing the as-built values the budget
compares against, both the mesh's and the new alignment ones, every one of which is currently
`UNSOURCED`; and a second mesh size, to turn the chip's fragility result into a trend. These are tracked as
[GitHub issues](https://github.com/roosado/photonn/issues).

## Install

Requires Python 3.11+ (developed on 3.12). From the repo root:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# POSIX:    source .venv/bin/activate
pip install -e ".[dev]"
```

## Test

```bash
pytest -q
```

Physics, layer, model, handoff, site and browser-cross-check tests pass (**`308 passed`**). The
checks that run the browser sources under Node — `test_asm_crosscheck.py`,
`test_d2nn_crosscheck.py`, `test_web_contract.py`, `test_error_widgets.py`,
`test_mount_queue.py` — require Node on `PATH` and skip cleanly if it is absent.

Four things the suite covers that a browser cannot, because the driven Chrome tab here is
always hidden and so never paints or lays out: the widget start-up gate
(`test_mount_queue.py`), the captions rendered from bundle provenance
(`test_web_contract.py`), the error widgets' layout at several widths and pixel ratios
(`test_error_widgets.py`), and the contents card's active-section mark
(`test_toc_spy.py`) — that tab scrolls without ever delivering a `scroll` event, which is
exactly the signal the mark is built on.

## The handoff

Python writes a single HDF5 file (trained parameters + geometry + operating point + frozen test
set + schema version). MATLAB reads it and models the imperfect device. The contract is specified
in [`docs/handoff_schema.md`](docs/handoff_schema.md); see `photonn/export.py` (writer) and
`photonn-hw/+io/read_handoff.m` (reader).

## Layout

```
photonn/        # Python design side (see CLAUDE.md for per-module responsibilities)
apps/           # diffraction_explorer.py (P1) · train_d2nn.py, visualize_d2nn.py (P2) · train_mesh.py, mesh_toolkit.py (P3) · build_site.py (site) · export_d2nn_web.py, d2nn_demo.py (browser classifier) · export_analogy_web.py, analogy_demo.py, analogy_figure.py (free-space↔chip correspondence) · export_mesh_web.py (trained chip → browser)
apps/web/       # dependency-free browser side: asm.js (propagation) · explorer.js (P1 widget) · d2nn.js, d2nn_demo.js, d2nn_stage.js, d2nn_weights.js (trained classifier + 3D stack) · errors.js (P4 error mechanisms, both architectures) · mesh_weights.js (trained chip, for the coupler widget) · scaling.js, optics_sweep.js (depth vs accuracy) · d2nn_compare.js (two models, one digit) · analogy.js, analogy_geom.js (P3 correspondence, demo only)
site/           # generated, self-contained, GitHub Pages ready: index.html (the live D²NN) · physics.html · chip.html · tolerance.html (the study) · optics.html
tests/          # pytest suite
docs/           # the written record, indexed in docs/README.md
photonn-hw/     # MATLAB as-built side (+io, +model, +meshmodel, +err, +mc, +viz, ErrorBudgetApp)
```
