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

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


def read_web_asset(name: str) -> str:
    """Return the text of an asset under ``apps/web`` (e.g. ``asm.js``)."""
    with open(os.path.join(WEB_DIR, name), "r", encoding="utf-8") as fh:
        return fh.read()


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
    return (
        "<script>\n"
        "  window.addEventListener('DOMContentLoaded', function () {\n"
        f"    window.PhotonnExplorer.mount(document.getElementById('{container_id}'), {cfg});\n"
        "  });\n"
        "</script>"
    )


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>photonn — live diffraction explorer</title>
<style>
  :root{{color-scheme:light dark;}}
  body{{margin:0;font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    background:#fff;color:#1b1f24;}}
  @media (prefers-color-scheme:dark){{body{{background:#0d1117;color:#e6eaf0;}}}}
  .wrap{{max-width:880px;margin:0 auto;padding:32px 22px 56px;}}
  h1{{font-size:1.5rem;margin:0 0 6px;}}
  .sub{{color:#5a6472;margin:0 0 22px;}}
  @media (prefers-color-scheme:dark){{.sub{{color:#9aa6b5;}}}}
  .note{{margin-top:26px;font-size:13px;color:#5a6472;border-top:1px solid #d7dde5;padding-top:14px;}}
  @media (prefers-color-scheme:dark){{.note{{color:#9aa6b5;border-color:#30363d;}}}}
  code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.9em;}}
</style>
</head>
<body>
<div class="wrap">
  <h1>Live diffraction explorer</h1>
  <p class="sub">Scalar diffraction by the band-limited angular-spectrum method, recomputed
  in your browser as you move the controls. Nothing is precomputed and nothing is fetched.</p>
  <div id="explorer"></div>
  <p class="note">The sampling flag compares the propagation distance <code>z</code> to the
  transfer-function critical distance <code>z_crit = N·dx²/λ</code>. Beyond it the angular
  spectrum under-samples; the band limit (Matsushima &amp; Shimobaba, 2009) keeps the result
  alias-free but drops high-angle content. Physics ported from
  <code>photonn.propagate.angular_spectrum</code> and verified against it to &lt; 1e-6.</p>
</div>
{bundle}
{mount}
</body>
</html>
"""


def build_html(**opts) -> str:
    """Return the full standalone explorer HTML string."""
    return _PAGE.format(bundle=explorer_bundle(), mount=explorer_mount(**opts))


def save_explorer(path: str = None, **opts) -> str:
    """Write the standalone explorer HTML and return its path."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "diffraction_explorer.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build_html(**opts))
    return path


def main():
    path = save_explorer()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
