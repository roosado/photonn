"""How big a built page is allowed to be, and why anyone should care.

These pages are self-contained by design: every figure and every trained model is
inlined, so there is nothing to fetch and they work from ``file://``. The cost is
that the document *is* the payload, and it grew until the site could not be
loaded on a phone at all -- ``optics.html`` reached 2.05 MB, of which 1.66 MB was
two weight bundles the browser had to tokenise as string literals before it could
paint anything.

Nothing in the build warns about that. The figures are hand-tuned constants and
the bundles are generated separately, so page weight is an emergent property of
edits nobody makes on purpose. These ceilings are the guard: they are generous
enough not to fire on ordinary prose edits, and tight enough that adding another
model or an unoptimised figure trips them.

If one fails, the fix is usually not to raise the ceiling. In order of leverage:
re-encode figures (``apps/build_site.py:encode_figure`` already keeps the
smallest of AVIF/WebP/PNG per figure), export a bundle at fewer bits
(``apps/web_bundle.py:encode_masks``), or check whether a bundle is inlined into
a page that does not use it.
"""
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "..", "site")

KB = 1024

#: page -> ceiling in KB. Measured after the 2026-08-10 redesign, with roughly 15%
#: headroom over what each page actually weighs.
BUDGET = {
    "index.html": 430,        # the live classifier (8-bit) + the 3D stage + two plates
    "physics.html": 70,       # prose + the diffraction explorer; the cheap one
    "chip.html": 150,         # the mesh topology plate; the analogy widget is gone
    "tolerance.html": 200,    # six error widgets sharing one mask, seven figures
    "optics.html": 1150,      # two models (one 56 masks at 4 bits) + the 56-mask budget
}

#: The whole site, as a reader walking the sequential path would meet it. Five
#: pages now rather than three, and the extra weight is real content -- the eight
#: candidate-L56 figures that make "depth costs tolerance" showable. It still
#: lands under the ceiling the three-page site was held to.
TOTAL_BUDGET_KB = 1900


def page(name):
    path = os.path.join(SITE, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not built; run python -m apps.build_site")
    return path


@pytest.mark.parametrize("name,ceiling", sorted(BUDGET.items()))
def test_page_is_within_budget(name, ceiling):
    size = os.path.getsize(page(name)) / KB
    assert size <= ceiling, (
        f"{name} is {size:.0f} KB against a {ceiling} KB ceiling. "
        "Re-encode the figures or export a bundle at fewer bits before raising this."
    )


def test_the_whole_site_is_within_budget():
    total = sum(os.path.getsize(page(n)) for n in BUDGET) / KB
    assert total <= TOTAL_BUDGET_KB, (
        f"the three pages total {total:.0f} KB against a {TOTAL_BUDGET_KB} KB ceiling"
    )


def test_pages_fetch_nothing():
    """The property that makes the size a budget rather than a first-load cost.

    If a page ever starts fetching, these ceilings stop describing what a visitor
    waits for, and the ``file://`` guarantee is gone with them.
    """
    for name in BUDGET:
        html = open(page(name), encoding="utf-8").read()
        for forbidden in ("fetch(", "XMLHttpRequest", "new Worker", "importScripts"):
            assert forbidden not in html, f"{name} contains {forbidden!r}; it is no longer self-contained"


@pytest.mark.parametrize("name", sorted(BUDGET))
def test_no_figure_ships_as_an_unoptimised_png(name):
    """PNG lost to AVIF on every figure in this project, often by 4x.

    A PNG data URI reappearing means encode_figure stopped being consulted -- a
    figure inlined by hand somewhere, most likely.
    """
    html = open(page(name), encoding="utf-8").read()
    assert "data:image/png;base64" not in html, (
        f"{name} inlines a PNG; encode_figure keeps the smallest of AVIF/WebP/PNG "
        "and PNG has never won here"
    )
