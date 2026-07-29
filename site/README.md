# photonn — project explainer page

`index.html` is a **single self-contained page** telling the whole photonn story: the
central fabrication-tolerance question, a section per phase (wave optics with a **live,
in-browser diffraction explorer**, the diffractive network, the MZI mesh, the error
budget), and "planned" placeholders for boson sampling and the mesh error budget.

Everything is inlined — CSS, JavaScript, and all figures as base64 data URIs — so the
page makes **no external requests** and works offline. The diffraction explorer runs a
hand-written FFT ported from `photonn.propagate.angular_spectrum` (cross-checked to
< 1e-6 by `tests/test_asm_crosscheck.py`).

## Regenerating

The page is generated, not hand-edited — edit `apps/build_site.py` (or the physics in
`apps/web/asm.js` / the widget in `apps/web/explorer.js`) and rebuild:

```
python -m apps.build_site
```

This writes two files:

- **`index.html`** — the full standalone document (this folder) for GitHub Pages.
- **`_artifact_body.html`** — a body-only variant used to publish the claude.ai
  Artifact (that host wraps the body in its own `<head>`/`<body>`). Not needed for Pages.

## Publishing to GitHub Pages (one-time, manual)

The project is not yet a git repository. To host this page:

```bash
# from the repo root (D:\Python\Photonn)
git init
git add .
git commit -m "photonn: source + project explainer site"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

Then on GitHub: **Settings → Pages → Build and deployment → Source: Deploy from a
branch**, branch `main`, folder `/site`. The page will be served at
`https://<you>.github.io/<repo>/`.

> `_artifact_body.html` is a build artifact; you can `.gitignore` it if you prefer to
> commit only `index.html`.
