"""Inline the multi-model comparison widget for embedding.

Mirrors :mod:`apps.d2nn_demo`, but carries *several* trained networks at once:
by default the 5-mask model and the 56-mask one from the optics sweep's
deliverable retrain. The 56-mask model's phases are 4-bit (see
:mod:`apps.web_bundle`), which is what keeps a network with eleven times the
parameters to under six times the download.

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

from apps import preview

from apps.diffraction_explorer import mount_script, read_web_asset

#: Weight bundles a board can carry: key -> (web asset, the window global it sets).
#: The globals are derived from the filenames by ``apps.web_bundle.js_global``.
#: Keys are internal handles; what a reader sees is the bundle's own ``label``,
#: which names a model by its depth.
BUNDLES = {
    "shipped": ("d2nn_weights.js", "D2NN_WEIGHTS"),
    "sweep": ("d2nn_sweep_weights.js", "D2NN_SWEEP_WEIGHTS"),
    "deep": ("d2nn_deep_weights.js", "D2NN_DEEP_WEIGHTS"),
}

#: The 5-mask model against the deepest trained one -- the sharpest contrast the
#: sweep supports, and the only pair whose two accuracies were measured the same
#: way. The first key is the board's baseline: it is captioned without a
#: comparison line, and the others are read against it. The 14-mask ``sweep``
#: bundle stays available for a fuller three-column telling of the sweep itself.
DEFAULT_MODELS = ("shipped", "deep")


def _assets(models):
    for key in models:
        if key not in BUNDLES:
            raise KeyError(f"unknown model {key!r}; known: {sorted(BUNDLES)}")
        yield BUNDLES[key]


def compare_bundle(models=DEFAULT_MODELS, stage: bool = False) -> str:
    """Return ``<script>`` tags with the chosen models, the engine and the widget.

    ``stage`` additionally inlines the 3D optical stage, for a board that draws
    one of its columns as a machine rather than only as a detector plane.
    """
    parts = [read_web_asset("asm.js")]
    parts += [read_web_asset(asset) for asset, _ in _assets(models)]
    parts += [
        read_web_asset("d2nn.js"),
        read_web_asset("digit_source.js"),
        read_web_asset("d2nn_compare.js"),
    ]
    if stage:
        parts.append(read_web_asset("d2nn_stage.js"))
    return "\n".join(f"<script>\n{p}\n</script>" for p in parts)


def compare_mount(container_id: str = "compare", models=DEFAULT_MODELS,
                  stage_id: str = None, stage_model: str = None, **opts) -> str:
    """Return a ``<script>`` that mounts the widget into ``#container_id``.

    ``models`` becomes a JavaScript array of the bundle globals, which is why it
    is spliced in rather than passed through ``json.dumps`` -- the widget wants
    the objects themselves, in column order.

    With ``stage_id``, the 3D stage is mounted there for the ``stage_model``
    column and fed the result **the board just computed**, via ``onResult``. Its
    network comes off the board too, so the model is built once and run once.

    Feeding it the digit instead would make the stage classify independently, and
    for the 56-mask column that is a second 57-hop forward pass -- about 130 ms --
    for a digit the board finished microseconds earlier. The stage still decides
    for itself *when to draw*; what it no longer does is recompute the physics.
    """
    keys = list(models)
    globals_js = ", ".join(f"window.{g}" for _, g in _assets(keys))
    cfg = ", ".join([f"models: [{globals_js}]"]
                    + [f"{k}: {json.dumps(v)}" for k, v in opts.items()])

    body = [f"var board = window.PhotonnD2NNCompare.mount(el, {{{cfg}}});"]
    if stage_id:
        if stage_model not in keys:
            raise KeyError(f"stage_model {stage_model!r} is not one of the board's models {keys}")
        idx = keys.index(stage_model)
        # The stage is scheduled from inside the board's own mount, as a second
        # job: it needs the board's network, and it sits well below the fold, so
        # it warms up as the reader comes down to it rather than at load.
        body += [
            f"var mountStage = function (sel) {{",
            f"  var stage = window.PhotonnD2NNStage.mount(sel, {{net: board.models[{idx}].net}});",
            f"  board.onResult(function (models, meta) {{ stage.setResult(models[{idx}].last, meta); }});",
            f"}};",
            f"if (window.PhotonnMount) window.PhotonnMount('{stage_id}', mountStage, {{defer: true}});",
            f"else mountStage(document.getElementById('{stage_id}'));",
        ]
    return mount_script(container_id, "\n".join(body))


#: What this preview page says about the widget it is showing.
PREVIEW = dict(
    title="two machines, one digit",
    heading="Two machines, one digit",
    standfirst="The 5-mask diffractive network and the sweep\u2019s 56-mask one, both running\n"
               "  live on the digit you pick.",
    hosts=("compare",),
)


def build_html(models=DEFAULT_MODELS, **opts) -> str:
    return preview.preview_page(
        bundle=compare_bundle(models, stage=bool(opts.get("stage_id"))),
        mount=compare_mount(**opts), **PREVIEW)


def save_demo(path: str = None, **opts) -> str:
    if path is None:
        path = preview.default_path(__file__, "compare_demo")
    return preview.save_preview(path, build_html(**opts))


def main():
    path = save_demo()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
