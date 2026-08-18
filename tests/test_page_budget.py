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
    # Raised from 70 when the in-page contents card landed. The card's CSS lives in
    # the one shared stylesheet, so every page pays for it whether or not it has
    # enough sections to need one -- and /physics, with three, is the page that
    # benefits least while paying the same 3 KB. At 70 it had 2.6 KB left, which is
    # a tripwire on the next paragraph rather than a guard on the payload. The
    # site-wide TOTAL_BUDGET_KB is the guard that actually catches bloat, and it is
    # unchanged with 60 KB spare.
    "physics.html": 80,       # prose + the diffraction explorer; the cheap one
    "chip.html": 150,         # the mesh topology plate; the analogy widget is gone
    # Raised from 200 when the chip's half of the error budget landed here rather
    # than on /chip -- in reading order the comparison only works once the stack's
    # budget is behind you. It costs this page the mesh bundle (8 KB of trained
    # phases) and the per-MZI sensitivity plate, and costs the site nothing on top,
    # since errors.js is now inlined once instead of on two pages.
    # Raised from 250 when the geometry half landed (issue #6): four more error
    # sources, the registration tolerance curve, and the joint-failure result. The
    # page is now the whole D2NN budget plus the chip comparison, which is what it
    # is for -- it is the study's destination page and the only one that grew.
    "tolerance.html": 290,    # ten error sources, seven widgets, nine figures
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
