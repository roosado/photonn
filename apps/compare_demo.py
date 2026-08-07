"""Inline the two-model comparison widget for embedding.

Mirrors :mod:`apps.d2nn_demo`, but carries *two* trained networks: the shipped
5-mask model and the sweep's 14-mask candidate. The candidate's phases are
8-bit (see :mod:`apps.export_sweep_web`), so the pair costs less than the shipped
model alone did as float32.

Run ``python -m apps.compare_demo`` to write ``compare_demo.html`` next to this
file -- a standalone page carrying just the comparison.
"""
from __future__ import annotations

import json
import os

from apps.diffraction_explorer import read_web_asset


def compare_bundle() -> str:
    """Return ``<script>`` tags with both models, the engine and the widget inlined."""
    parts = [
        read_web_asset("asm.js"),
        read_web_asset("d2nn_weights.js"),
        read_web_asset("d2nn_sweep_weights.js"),
        read_web_asset("d2nn.js"),
        read_web_asset("d2nn_compare.js"),
    ]
    return "\n".join(f"<script>\n{p}\n</script>" for p in parts)


def compare_mount(container_id: str = "compare", **opts) -> str:
    """Return a ``<script>`` that mounts the widget into ``#container_id``."""
    cfg = json.dumps(opts)
    return (
        "<script>\n"
        "  window.addEventListener('DOMContentLoaded', function () {\n"
        f"    window.PhotonnD2NNCompare.mount(document.getElementById('{container_id}'), {cfg});\n"
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


def build_html(**opts) -> str:
    return _PAGE.format(bundle=compare_bundle(), mount=compare_mount(**opts))


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
