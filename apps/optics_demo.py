"""Inline the optics-sweep widget for embedding.

Mirrors :mod:`apps.analogy_demo`: bundles ``apps/web/optics_sweep.js`` (the
measured accuracies, written by :mod:`apps.sweep_report`) and
``apps/web/optics.js`` into inline ``<script>`` tags, so the explainer page stays
CSP-safe and offline. The widget recomputes its geometry live and loads no
weights, so the bundle is small.

Run ``python -m apps.optics_demo`` to write ``optics_demo.html`` next to this
file -- a standalone page carrying just this figure.
"""
from __future__ import annotations

import json
import os

from apps.diffraction_explorer import read_web_asset


def optics_bundle() -> str:
    """Return ``<script>`` tags with the sweep data + widget inlined."""
    parts = [read_web_asset("optics_sweep.js"), read_web_asset("optics.js")]
    return "\n".join(f"<script>\n{p}\n</script>" for p in parts)


def optics_mount(container_id: str = "optics", **opts) -> str:
    """Return a ``<script>`` that mounts the widget into ``#container_id``.

    ``opts`` are forwarded to ``PhotonnOptics.mount`` (currently just ``zMm``,
    the initial separation).
    """
    cfg = json.dumps(opts)
    return (
        "<script>\n"
        "  window.addEventListener('DOMContentLoaded', function () {\n"
        f"    window.PhotonnOptics.mount(document.getElementById('{container_id}'), {cfg});\n"
        "  });\n"
        "</script>"
    )


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>photonn &mdash; what separation buys</title>
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
  <h1>What separation buys</h1>
  <p class="sub">The diffractive network is capacity-limited, not data-limited. Its remaining
  levers are optical &mdash; and the first one is simply how far apart the phase masks sit.</p>
  <div id="optics"></div>
</div>
{bundle}
{mount}
</body>
</html>
"""


def build_html(**opts) -> str:
    return _PAGE.format(bundle=optics_bundle(), mount=optics_mount(**opts))


def save_demo(path: str = None, **opts) -> str:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "optics_demo.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build_html(**opts))
    return path


def main():
    path = save_demo(zMm=3)
    print(f"wrote {path} ({os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    main()
