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

from apps import preview

from apps.diffraction_explorer import mount_script, read_web_asset


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
    return mount_script(container_id, f"window.PhotonnOptics.mount(el, {cfg});")


#: What this preview page says about the widget it is showing.
PREVIEW = dict(
    title="what separation buys",
    heading="What separation buys",
    standfirst="The diffractive network is capacity-limited, not data-limited. Its remaining\n"
               "  levers are optical &mdash; and the first one is simply how far apart the phase masks sit.",
    hosts=("optics",),
)


def build_html(**opts) -> str:
    return preview.preview_page(bundle=optics_bundle(), mount=optics_mount(**opts),
                                **PREVIEW)


def save_demo(path: str = None, **opts) -> str:
    if path is None:
        path = preview.default_path(__file__, "optics_demo")
    return preview.save_preview(path, build_html(**opts))


def main():
    path = save_demo(zMm=3)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
