# photonn — the generated site

Five pages, each a **single self-contained HTML document**: CSS, JavaScript and every figure are
inlined (figures as base64 data URIs), so they make **no external requests** and work offline —
including straight off `file://`.

They read in order, and the topbar lists all five:

1. **`index.html`** — *This neural network is made of light.* The trained D²NN running its
   forward pass in your browser: pick or draw a digit, watch it cross five phase masks onto ten
   detectors, in a **3D view of the optical stack** and as exact per-plane frames. Then what a
   phase mask actually computes, why anyone would build a computer out of light, and why the
   network is wrong about one digit in four — with the **confusion matrix of the ideal model**
   showing which digits it actually trades (5→3, 8→3, 9↔4).
2. **`physics.html`** — the angular-spectrum propagator, the sampling limit `z_crit` that bounds
   it, the **live diffraction explorer** with a reading guide for its controls, and the proof
   that the whole stack collapses to one linear operator followed by a single `|E|²`.
3. **`chip.html`** — the MZI mesh, the crosswalk table, and why a phase mask *is* a column of
   phase shifters. Then **what breaks it**: the chip's own half of the error budget, where the
   two machines stop being the same machine. It needs its delays ten times more accurate than the
   glass stack does, because 72 columns of interferometers in series carry each other's errors
   forward, and it has a constraint the stack cannot have at all — couplers that split unevenly,
   on the slider. A quarter of one mesh turns out to need no fabrication tolerance whatever. The
   interactive correspondence widget was removed because the table already made its argument.
4. **`tolerance.html`** — the fabrication error budget, as **one section per error source**.
   Each carries a purpose-built widget showing what that error physically does, its measured
   tolerance curve, and the one number that matters. Closes on the ranking: five sources are
   comfortable, crosstalk fails by 4×, and geometry is not modelled at all.
5. **`optics.html`** — what scaling buys, and where it stops. The depth-vs-accuracy chart, the
   5-mask model running beside the 56-mask one on one digit, what the
   extra masks cost in tolerance, and then the wall: a mask stack is one linear operator, so
   depth converges rather than compounds, and the way through is a nonlinearity.

Plus **`_artifact_body.html`** — a body-only variant of the front page for publishing as a
claude.ai Artifact (that host supplies its own `<head>`/`<body>`, and needs absolute links since
it has no sibling files). Gitignored; not part of the deployed site.

> `classifier.html` **was deleted** in the 2026-08-10 redesign. Its reader-facing half became the
> front page and its methodology moved to `physics.html`. `tests/test_site_links.py` fails if
> anything links to it again.

None of the physics is precomputed. The browser runs a hand-written FFT ported from
`photonn.propagate.angular_spectrum` (`apps/web/asm.js`, cross-checked to < 1e-6 by
`tests/test_asm_crosscheck.py`), and the classifier's predictions match the PyTorch model
exactly (`tests/test_d2nn_crosscheck.py`).

## Widgets

| File | Page | What it does |
|---|---|---|
| `d2nn.js` + `d2nn_demo.js` | index | the live classifier |
| `d2nn_stage.js` | index, optics | the 3D optical stack |
| `explorer.js` | physics | the diffraction explorer |
| `errors.js` | tolerance, chip | **seven** error-mechanism widgets, one per source |
| `mesh_weights.js` | chip | the trained chip's 2,628 settings, 16-bit, for `errors.js` |
| `scaling.js` | optics | accuracy against mask count |
| `d2nn_compare.js` | optics | 5 masks vs 56 masks, one digit |
| `analogy.js` | *(none)* | kept for `apps/analogy_demo.py`; the site no longer mounts it |

`errors.js` is **mechanism only**: it never runs a classifier and computes no accuracy, so
it cannot contradict the measured curve beside it. Both models it draws are the real thing. The
128² phase mask is cut out of `d2nn_weights.js` at build time by
`build_site.error_mask_bundle()` — one copy of the trained phases in the repo, and this is a
slice of it. The chip comes from `mesh_weights.js`, written by `apps.export_mesh_web`, and
`tests/test_mesh_web.py` rebuilds the operator from it under Node and checks it against
`photonn.mzi`, because a transposed index would draw a confident picture of a different chip.

Six kinds mount on `/tolerance` in page order `crosstalk, phase, detector, loss, wavelength,
quant`; the seventh, `mesh`, mounts on `/chip`. The whole file is inlined on both pages rather
than split, because all seven share a stylesheet and a second copy would mean a second
`STYLE_ID` — a bug this repo has shipped once. They split into two shapes with **two layout
rules that are load-bearing** (both were broken and fixed in `cd372b1`):

* **Five triptychs** (crosstalk, phase, wavelength, quant, mesh) — three square panels reading
  as-designed / as-built / the difference. They must **never wrap**, because sweeping the
  slider only shows the mechanism if all three are on screen at once. They were
  `flex: 1 1 150px; min-width: 132px`, which needs 420 px of row; a phone column is about
  310 px, so the third panel wrapped, the rest grew to full width, and the reader got one
  enormous picture at a time. Now `flex: 1 1 0; min-width: 0` on a row that cannot wrap, so
  three always fit and simply shrink — about 95 px each at 300 px of column.
* **Two plots** (loss, detector) — one wide pane, capped at 640 px by `.ex-pane.ex-wide`.
  A plot must **measure its pane before it draws**. These used a fixed 420-unit drawing space
  with an inline height and let `width:100%` scale the bitmap to whatever the column really
  was, which scales x without scaling y: flattened on a desktop, stretched tall on a phone,
  labels and all. `fitCanvas()` now measures first and draws one unit per CSS pixel, and
  `onWidthChange()` redraws on a real width change (guarded on the measured width, since the
  redraw sets the canvas height and would otherwise answer its own observer forever).

> **The stylesheet is a JS template literal, so it must contain no backticks.** One anywhere
> inside it — including inside a CSS comment — terminates the literal and breaks the whole
> widget. This has happened once; the Node runner above is what caught it.

Neither rule can be checked in a driven browser — that tab is always hidden, so it never lays
anything out. `tests/error_widget_runner.js` mounts all seven kinds against a DOM stand-in at
300/480/1042 px and device pixel ratios 1 and 2, and `tests/test_error_widgets.py` asserts that
every canvas's **bitmap aspect equals the aspect it is displayed at**, plus the stylesheet rules
above. The runner's mini-flexbox **parses its rules out of `errors.js`'s own CSS** rather than
restating them, so "three panels stay on one row" tests the stylesheet that ships and not the
test file.

## Maths is MathML

Expressions are MathML, which every current browser renders natively, so real notation costs
no library. `apps/build_site.py` holds a compact token notation (`mrow`, `mfrac`, `msqrt`)
expanded into a `MATH` dict and substituted as `@@MATH_*@@`, mirroring `@@FIG_*@@`. Code and
UI identifiers keep the `.q` mono pill; only mathematics becomes MathML.

> **Never set CSS `display` or `overflow` on a `<math>` element.** MathML lays out as
> `display: math`/`block math`, and overriding it drops the element into ordinary CSS block
> layout, which puts every child on its own line — a nine-term operator product renders as
> nine stacked rows. The scroll container lives on the `.eq` wrapper instead.

## Regenerating

These pages are **generated, not hand-edited** — edit `apps/build_site.py`, or the browser
sources under `apps/web/`, and rebuild:

```bash
python -m apps.build_site      # writes all six files above
```

The build is deterministic: running it twice produces byte-identical output.

Navigation is generated from one list. `apps.build_site.PAGES` holds every page's filename, nav
label, title and hand-off blurb; the topbar, the sequential "next" card and the relative →
absolute link swap the Artifact needs are all derived from it. **Adding or renaming a page means
editing `PAGES`, not the markup.** Links between pages are written as `@@HREF_<key>@@` tokens and
resolved last, which is what lets one rendered body be emitted twice.

If the **trained model** changes, regenerate in this order before rebuilding the site:

```bash
python -m apps.export_d2nn_web      # -> apps/web/d2nn_weights.js, tests/fixtures/d2nn_reference.json
python -m apps.export_analogy_web   # -> apps/web/analogy_geom.js
python -m apps.export_mesh_web      # -> apps/web/mesh_weights.js   (needs handoff schema 0.2.0)
python -m apps.analogy_figure       # -> docs/figures/phase3_correspondence.png
python -m apps.build_site
```

If the **optics sweep** is re-run, regenerate its bundle too — the depth-vs-accuracy chart
reads its points from there rather than carrying its own copy:

```bash
python -m apps.sweep_report         # -> apps/web/optics_sweep.js, docs/figures/optics_sweep.png
```

`apps/web/d2nn_weights.js`, `apps/web/analogy_geom.js` and `apps/web/mesh_weights.js` are
**committed on purpose**: `.gitignore` excludes `*.h5`/`*.pt`, so they are the repo's only copies
of what the trained models say, and the only way the site rebuilds from a fresh clone.

The standalone one-widget pages (`apps/d2nn_demo.html`, `apps/diffraction_explorer.html`,
`apps/analogy_demo.html`, `apps/compare_demo.html`) are built by `python -m apps.d2nn_demo`,
`python -m apps.diffraction_explorer`, `python -m apps.analogy_demo` and
`python -m apps.compare_demo` respectively — handy for eyeballing a single widget without
rebuilding the whole site.

## Quoted figures are hand-maintained

**Nothing checks the numbers on these pages against the model.** They are Python string
constants in `apps/build_site.py`, and they have gone stale before — the tolerance edges on the
live site were the pre-retrain values for four months. After any retrain or any re-run of the
error budget, re-grep `build_site.py` for the accuracies, the power budget *and* the tolerance
edges, not just the headline — on `/chip` as well as `/tolerance` now, since both carry measured
limits. `docs/tolerance_d2nn.md`, `docs/tolerance_mesh.md` and `docs/phase2_dnn.md` are the
sources of record.

## Publishing

`.github/workflows/pages.yml` deploys this folder to GitHub Pages on every push to `main`
(GitHub Actions Pages source, so an arbitrary subfolder can be served — the branch-deploy UI
only offers `/` or `/docs`). Live at **https://roosado.github.io/photonn/**.

> **The workflow only publishes `site/`; it never runs a build.** Nothing regenerates these
> pages in CI, so they must be rebuilt locally and committed whenever their sources change, or
> the deployed site will silently lag the repo.
