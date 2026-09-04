"""Interactive diffraction explorer -- Phase 1 deliverable (live, browser-side).

This builds a self-contained HTML page that computes scalar diffraction by the
angular-spectrum method **live in the browser** -- every control is continuous
(aperture shape/size, distance, wavelength, grid size), and the sampling-violation
flag (``z`` vs ``z_crit = n*dx^2/lam``) updates in real time.

The physics is the JavaScript port in ``apps/web/asm.js`` (a faithful translation
of :func:`photonn.propagate.angular_spectrum`, checked against it to < 1e-6 by
``tests/test_asm_crosscheck.py``); the UI widget is ``apps/web/explorer.js``. This
module just inlines both into a standalone page -- so there is no server, no
network, and no precomputation (the previous Plotly version could only flip between
prebaked distance frames). The same inlined bundle is reused by ``apps/build_site.py``
for the project explainer page.

Run ``python -m apps.diffraction_explorer`` to write ``diffraction_explorer.html``
next to this file.
"""
from __future__ import annotations

import json
import os

from apps import preview

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


def read_web_asset(name: str) -> str:
    """Return the text of an asset under ``apps/web`` (e.g. ``asm.js``)."""
    with open(os.path.join(WEB_DIR, name), "r", encoding="utf-8") as fh:
        return fh.read()


def mount_queue_bundle() -> str:
    """Return the ``<script>`` holding the page's mount scheduler.

    Include this once per multi-widget page, before any mount script. Pages that
    carry only one widget can omit it -- :func:`mount_script` falls back to a
    plain listener when the scheduler is absent, which is what the standalone
    demo pages under ``apps/`` rely on.
    """
    return f"<script>\n{read_web_asset('mount_queue.js')}\n</script>\n"


def mount_script(container_id: str, body: str, defer: bool = False) -> str:
    """Wrap widget-mount JavaScript so the page decides when it runs.

    ``body`` is JavaScript that mounts the widget into the local variable ``el``.
    It is handed to ``apps/web/mount_queue.js``, which holds every widget until
    after the first paint and then starts them one at a time -- so a page carrying
    several expensive widgets never blocks the main thread on all of them at once.
    With ``defer``, the widget additionally waits until the reader is approaching
    it. Either way it still starts by itself: nothing here asks for a tap.

    The ``readyState`` check in the fallback matters for nested mounts. A widget
    that schedules another one (the optics board mounting the 3D stage off its own
    network) runs long after DOMContentLoaded, so a fallback that only ever added
    a listener would silently never fire.
    """
    ident = json.dumps(container_id)
    opts = ", {defer: true}" if defer else ""
    lines = ["<script>", "  (function () {", "    function boot(el) {"]
    lines += [f"      {line}" for line in body.strip().splitlines()]
    lines += [
        "    }",
        f"    if (window.PhotonnMount) window.PhotonnMount({ident}, boot{opts});",
        "    else if (document.readyState === 'loading') {",
        f"      window.addEventListener('DOMContentLoaded', function () {{ boot(document.getElementById({ident})); }});",
        "    } else {",
        f"      boot(document.getElementById({ident}));",
        "    }",
        "  })();",
        "</script>",
    ]
    return "\n".join(lines)


def explorer_bundle() -> str:
    """Return ``<script>`` tags with asm.js + explorer.js inlined (CSP-safe, offline).

    Shared by the standalone explorer and the explainer page so both embed exactly
    the same, test-verified physics.
    """
    asm = read_web_asset("asm.js")
    explorer = read_web_asset("explorer.js")
    return f"<script>\n{asm}\n</script>\n<script>\n{explorer}\n</script>"


def explorer_mount(container_id: str = "explorer", **opts) -> str:
    """Return a ``<script>`` that mounts the widget into ``#container_id``.

    ``opts`` are SI-unit overrides forwarded to ``PhotonnExplorer.mount`` (extent,
    grid, wavelength, apertureSize, distance, shape).
    """
    cfg = json.dumps(opts)
    return mount_script(container_id, f"window.PhotonnExplorer.mount(el, {cfg});")


#: What this preview page says about the widget it is showing.
PREVIEW = dict(
    title="live diffraction explorer",
    heading="Live diffraction explorer",
    standfirst="Scalar diffraction by the band-limited angular-spectrum method, recomputed\n"
               "  in your browser as you move the controls. Nothing is precomputed and nothing is fetched.",
    hosts=("explorer",),
    note="The sampling flag compares the propagation distance <code>z</code> to the\n"
         "  transfer-function critical distance <code>z_crit = N\u00b7dx\u00b2/\u03bb</code>. Beyond it the angular\n"
         "  spectrum under-samples; the band limit (Matsushima &amp; Shimobaba, 2009) keeps the result\n"
         "  alias-free but drops high-angle content. Physics ported from\n"
         "  <code>photonn.propagate.angular_spectrum</code> and verified against it to &lt; 1e-6.",
)


def build_html(**opts) -> str:
    """Return the full standalone explorer HTML string."""
    return preview.preview_page(bundle=explorer_bundle(), mount=explorer_mount(**opts),
                                **PREVIEW)


def save_explorer(path: str = None, **opts) -> str:
    """Write the standalone explorer HTML and return its path."""
    if path is None:
        path = preview.default_path(__file__, "diffraction_explorer")
    return preview.save_preview(path, build_html(**opts))


def main():
    path = save_explorer()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
