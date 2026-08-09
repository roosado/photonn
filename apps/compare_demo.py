"""Inline the multi-model comparison widget for embedding.

Mirrors :mod:`apps.d2nn_demo`, but carries *several* trained networks at once:
by default the shipped 5-mask model and the 56-mask candidate from the optics
sweep's deliverable retrain. The candidate's phases are 8-bit (see
:mod:`apps.web_bundle`), which is what keeps a network with eleven times the
parameters to under three times the download.

Which bundles a board carries is a page-level choice, so it is a parameter here
rather than a fact baked into the widget: ``models`` names keys of
:data:`BUNDLES`, and both the inlined ``<script>`` tags and the mount call are
built from that one list. A board can be given two models or four without
either file knowing anything about them.

All of them are fed by one ``digit_source.js`` -- the gallery and draw pad are a
separate widget precisely so the models cannot be shown different inputs.

Run ``python -m apps.compare_demo`` to write ``compare_demo.html`` next to this
file -- a standalone page carrying just the comparison.
"""
from __future__ import annotations

import json
import os

from apps.diffraction_explorer import read_web_asset

#: Weight bundles a board can carry: key -> (web asset, the window global it sets).
#: The globals are derived from the filenames by ``apps.web_bundle.js_global``.
BUNDLES = {
    "shipped": ("d2nn_weights.js", "D2NN_WEIGHTS"),
    "sweep": ("d2nn_sweep_weights.js", "D2NN_SWEEP_WEIGHTS"),
    "deep": ("d2nn_deep_weights.js", "D2NN_DEEP_WEIGHTS"),
}

#: The shipped model against the deepest trained one -- the sharpest contrast the
#: sweep supports, and the only pair whose two accuracies were measured the same
#: way. The 14-mask ``sweep`` bundle stays available for a fuller three-column
#: telling of the sweep itself.
DEFAULT_MODELS = ("shipped", "deep")


def _assets(models):
    for key in models:
        if key not in BUNDLES:
            raise KeyError(f"unknown model {key!r}; known: {sorted(BUNDLES)}")
        yield BUNDLES[key]


def compare_bundle(models=DEFAULT_MODELS) -> str:
    """Return ``<script>`` tags with the chosen models, the engine and the widget."""
    parts = [read_web_asset("asm.js")]
    parts += [read_web_asset(asset) for asset, _ in _assets(models)]
    parts += [
        read_web_asset("d2nn.js"),
        read_web_asset("digit_source.js"),
        read_web_asset("d2nn_compare.js"),
    ]
    return "\n".join(f"<script>\n{p}\n</script>" for p in parts)


def compare_mount(container_id: str = "compare", models=DEFAULT_MODELS, **opts) -> str:
    """Return a ``<script>`` that mounts the widget into ``#container_id``.

    ``models`` becomes a JavaScript array of the bundle globals, which is why it
    is spliced in rather than passed through ``json.dumps`` -- the widget wants
    the objects themselves, in column order.
    """
    globals_js = ", ".join(f"window.{g}" for _, g in _assets(models))
    cfg = ", ".join([f"models: [{globals_js}]"]
                    + [f"{k}: {json.dumps(v)}" for k, v in opts.items()])
    return (
        "<script>\n"
        "  window.addEventListener('DOMContentLoaded', function () {\n"
        f"    window.PhotonnD2NNCompare.mount(document.getElementById('{container_id}'), "
        f"{{{cfg}}});\n"
        "  });\n"
        "</script>"
    )


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>photonn &mdash; two machines, one digit</title>
<style>
  :root{{color-scheme:light dark;}}
  body{{margin:0;font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    background:#fff;color:#1b1f24;}}
  @media (prefers-color-scheme:dark){{body{{background:#0d1117;color:#e6eaf0;}}}}
  .wrap{{max-width:880px;margin:0 auto;padding:32px 22px 56px;}}
  h1{{font-size:1.5rem;margin:0 0 6px;}}
  .sub{{color:#5a6472;margin:0 0 22px;}}
  @media (prefers-color-scheme:dark){{.sub{{color:#9aa6b5;}}}}
</style>
</head>
<body>
<div class="wrap">
  <h1>Two machines, one digit</h1>
  <p class="sub">The shipped diffractive network and the sweep's deeper candidate, both running
  live on the digit you pick.</p>
  <div id="compare"></div>
</div>
{bundle}
{mount}
</body>
</html>
"""


def build_html(models=DEFAULT_MODELS, **opts) -> str:
    return _PAGE.format(bundle=compare_bundle(models),
                        mount=compare_mount(models=models, **opts))


def save_demo(path: str = None, **opts) -> str:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "compare_demo.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build_html(**opts))
    return path


def main():
    path = save_demo()
    print(f"wrote {path} ({os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    main()
