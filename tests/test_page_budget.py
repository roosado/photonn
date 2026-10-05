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

from apps.build_site import PAGES
from built_site import page_html

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "..", "site")

KB = 1024

#: page -> ceiling in KB. Measured after the 2026-08-10 redesign, with roughly 15%
#: headroom over what each page actually weighs.
CEILING_KB = {
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
    # Measured 216 KB when it landed (Phase 5): four figures 93 KB, the activation
    # widget 25, the trained chip at 16 bits 24, mesh.js 7, the shared chrome and
    # prose the rest. Ceiling at the file's usual ~15 % over.
    "activation.html": 250,   # the device curve + the live two-layer chip + four figures
}

#: The whole site, as a reader walking the sequential path would meet it. Six
#: pages now rather than three, and the extra weight is real content -- the eight
#: candidate-L56 figures that make "depth costs tolerance" showable.
#:
#: Raised from 1900 when the widgets' shared canvas code moved into
#: ``apps/web/plot.js``. That is a deliberate trade and it goes the wrong way on
#: this metric: the module is 12 KB and lands on the four pages that draw
#: anything, while the per-widget copies it replaced came to about 7 KB in total.
#: Net +42 KB, 2.2% of the site.
#:
#: Taken anyway, because the duplication was not free either. Nine widgets each
#: held their own dpr clamp, canvas resize, palette read and colour ramp, and the
#: copies had drifted: one of them reallocated a ~1440x860 bitmap on every
#: animation frame, and three learned about theme changes from a source this
#: site's own toggle never fires. Those are the failures a shared module makes
#: impossible, and 42 KB is what they cost to prevent.
#:
#: Every per-page ceiling above is unchanged and still passes. /chip gets no copy
#: at all -- it carries no canvas, and ``Page.widgets`` is what makes that
#: answerable.
#:
#: Raised from 1960 to 2210 when the sixth page landed (Phase 5, the activation):
#: its measured 216 KB plus the same ~15 % every per-page ceiling carries. The site
#: had 46 KB of headroom, so a sixth page could not fit under the old figure; that
#: is the one ceiling the page was allowed to raise, and no other moved with it.
TOTAL_BUDGET_KB = 2210


#: Every page a reader can reach, taken from PAGES rather than restated. The
#: ceilings above are per-page judgements and stay hand-written; the *set* of
#: pages is not a judgement, and keeping a second copy of it here meant a page
#: added to PAGES was link-checked and index-checked but silently escaped the
#: weight, self-containment and figure-encoding guards below.
PAGE_FILES = tuple(p.file for p in PAGES)


def test_every_page_has_a_ceiling():
    assert set(CEILING_KB) == set(PAGE_FILES), (
        "CEILING_KB and PAGES disagree about which pages exist. A page without a "
        "ceiling is a page with no weight guard at all."
    )


def size_kb(name):
    return len(page_html(name).encode("utf-8")) / KB


@pytest.mark.parametrize("name,ceiling", sorted(CEILING_KB.items()))
def test_page_is_within_budget(name, ceiling):
    size = size_kb(name)
    assert size <= ceiling, (
        f"{name} is {size:.0f} KB against a {ceiling} KB ceiling. "
        "Re-encode the figures or export a bundle at fewer bits before raising this."
    )


def test_the_whole_site_is_within_budget():
    total = sum(size_kb(n) for n in PAGE_FILES)
    assert total <= TOTAL_BUDGET_KB, (
        f"the {len(PAGE_FILES)} pages total {total:.0f} KB against a {TOTAL_BUDGET_KB} KB ceiling"
    )


@pytest.mark.parametrize("name", PAGE_FILES)
def test_pages_fetch_nothing(name):
    """The property that makes the size a budget rather than a first-load cost.

    If a page ever starts fetching, these ceilings stop describing what a visitor
    waits for, and the ``file://`` guarantee is gone with them.
    """
    html = page_html(name)
    for forbidden in ("fetch(", "XMLHttpRequest", "new Worker", "importScripts"):
        assert forbidden not in html, f"{name} contains {forbidden!r}; it is no longer self-contained"


@pytest.mark.parametrize("name", PAGE_FILES)
def test_no_figure_ships_as_an_unoptimised_png(name):
    """PNG lost to AVIF on every figure in this project, often by 4x.

    A PNG data URI reappearing means encode_figure stopped being consulted -- a
    figure inlined by hand somewhere, most likely.
    """
    html = page_html(name)
    assert "data:image/png;base64" not in html, (
        f"{name} inlines a PNG; encode_figure keeps the smallest of AVIF/WebP/PNG "
        "and PNG has never won here"
    )
