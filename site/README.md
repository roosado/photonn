# photonn — the generated site

Five pages, each a **single self-contained HTML document**: CSS, JavaScript and every figure are
inlined (figures as base64 data URIs), so they make **no external requests** and work offline —
including straight off `file://`.

They read in order, and the topbar lists all five:

1. **`index.html`** — *This neural network is made of light.* The trained D²NN running its
   forward pass in your browser: pick or draw a digit, watch it cross five phase masks onto ten
   detectors, in a **3D view of the optical stack** and as exact per-plane frames. Then what a
   phase mask actually computes, why anyone would build a computer out of light, and why the
   network is wrong about one digit in four.
2. **`physics.html`** — the angular-spectrum propagator, the sampling limit `z_crit` that bounds
   it, the **live diffraction explorer**, and the proof that the whole stack collapses to one
   linear operator followed by a single `|E|²`.
3. **`chip.html`** — the MZI mesh, and the **free-space ↔ chip correspondence** figure. Why a
   phase mask *is* a column of phase shifters, and the one number the two machines differ on.
4. **`tolerance.html`** — the fabrication error budget. The binding constraint, the seven
   tolerance curves and the sensitivity map, plus what a deeper network costs in tolerance to
   buy its accuracy.
5. **`optics.html`** — live work: the optics sweep, and the shipped 5-mask model running beside
   the unshipped 56-mask candidate on one digit.

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
python -m apps.analogy_figure       # -> docs/figures/phase3_correspondence.png
python -m apps.build_site
```

`apps/web/d2nn_weights.js` and `apps/web/analogy_geom.js` are **committed on purpose**:
`.gitignore` excludes `*.h5`/`*.pt`, so they are the repo's only copies of what the trained
models say, and the only way the site rebuilds from a fresh clone.

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
edges, not just the headline. `docs/tolerance_d2nn.md` and `docs/phase2_dnn.md` are the sources
of record.

## Publishing

`.github/workflows/pages.yml` deploys this folder to GitHub Pages on every push to `main`
(GitHub Actions Pages source, so an arbitrary subfolder can be served — the branch-deploy UI
only offers `/` or `/docs`). Live at **https://roosado.github.io/photonn/**.

> **The workflow only publishes `site/`; it never runs a build.** Nothing regenerates these
> pages in CI, so they must be rebuilt locally and committed whenever their sources change, or
> the deployed site will silently lag the repo.
