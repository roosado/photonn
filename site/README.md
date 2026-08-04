# photonn — the generated site

Two pages, both **single self-contained HTML documents**: CSS, JavaScript and every figure are
inlined (figures as base64 data URIs), so they make **no external requests** and work offline.

- **`index.html`** — the project explainer: the central fabrication-tolerance question and a
  section per phase, with two live widgets — the Phase-1 **diffraction explorer** and the
  Phase-3 **free-space ↔ chip correspondence** figure — plus "planned" placeholders for boson
  sampling and the mesh error budget.
- **`classifier.html`** — the trained D²NN running its forward pass in your browser: pick or
  draw a digit, watch it cross five phase masks onto ten detectors, in a **3D view of the
  optical stack** and as exact per-plane frames.
- **`_artifact_body.html`** — a body-only variant of the explainer for publishing as a
  claude.ai Artifact (that host supplies its own `<head>`/`<body>`). Gitignored; not part of
  the deployed site.

None of the physics is precomputed. The browser runs a hand-written FFT ported from
`photonn.propagate.angular_spectrum` (`apps/web/asm.js`, cross-checked to < 1e-6 by
`tests/test_asm_crosscheck.py`), and the classifier's predictions match the PyTorch model
exactly (`tests/test_d2nn_crosscheck.py`).

## Regenerating

These pages are **generated, not hand-edited** — edit `apps/build_site.py`, or the browser
sources under `apps/web/`, and rebuild:

```bash
python -m apps.build_site      # writes all three files above
```

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
`apps/analogy_demo.html`) are built by `python -m apps.d2nn_demo`,
`python -m apps.diffraction_explorer` and `python -m apps.analogy_demo` respectively — handy for
eyeballing a single widget without rebuilding the whole site.

## Publishing

`.github/workflows/pages.yml` deploys this folder to GitHub Pages on every push to `main`
(GitHub Actions Pages source, so an arbitrary subfolder can be served — the branch-deploy UI
only offers `/` or `/docs`). Live at **https://roosado.github.io/photonn/**.

> **The workflow only publishes `site/`; it never runs a build.** Nothing regenerates these
> pages in CI, so `index.html` and `classifier.html` must be rebuilt locally and committed
> whenever their sources change, or the deployed site will silently lag the repo.
