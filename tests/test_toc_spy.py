"""The contents card's active-section mark, checked where a browser cannot check it.

The card highlights the section being read. That is the one part of this feature a
screenshot cannot confirm: in the driven Chrome tab the page really does scroll
(``window.scrollY`` changes) but **no ``scroll`` event is ever delivered, and no
``IntersectionObserver`` callback fires either** -- measured on this page, not
assumed. An earlier version of the mark was built on an observer and looked fine in
a screenshot while never updating at all.

So it runs under Node against a stand-in for the parts of the DOM it touches, with
scroll position driven by hand, and the script is lifted **out of the built page**
rather than retyped -- these assertions are about the bytes a reader downloads. See
``tests/toc_spy_runner.js``.

What it pins:

* the mark is ``aria-current="location"``, never ``"page"`` -- ``test_site_links.py``
  asserts exactly one ``aria-current="page"`` anchor per page and that it is the
  topbar's own;
* exactly one entry is marked at any scroll position, including before any
  scrolling has happened;
* the rule is *the last heading scrolled past*, not *the heading on screen*, because
  sections here are hundreds of pixels long and a rule based on visibility leaves
  the card blank for most of a page.
"""
import json
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "toc_spy_runner.js")
PAGE = os.path.join(os.path.dirname(HERE), "site", "tolerance.html")

node = shutil.which("node")
pytestmark = [
    pytest.mark.skipif(node is None, reason="node not on PATH; contents-card checks skipped"),
    pytest.mark.skipif(not os.path.exists(PAGE), reason="site not built"),
]


@pytest.fixture(scope="module")
def out():
    proc = subprocess.run([node, RUNNER], capture_output=True, text=True)
    assert proc.returncode == 0, f"contents-card runner failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def test_a_section_is_marked_before_any_scrolling(out):
    """Opening the page must not leave the card with nothing marked."""
    assert out["onLoad"]["count"] == 1
    assert out["onLoad"]["id"] == out["ids"][0]


def test_the_mark_follows_the_last_heading_scrolled_past(out):
    wrong = [w for w in out["walk"] if w["expected"] != w["got"]]
    assert not wrong, f"card marked the wrong section at {wrong}"


def test_exactly_one_entry_is_ever_marked(out):
    counts = {w["count"] for w in out["walk"]}
    counts |= {out["onLoad"]["count"], out["afterJump"]["count"], out["backToTop"]["count"]}
    assert counts == {1}


def test_the_mark_never_claims_to_be_the_current_page(out):
    """`aria-current="page"` belongs to the topbar and to exactly one anchor."""
    values = {w["value"] for w in out["walk"]} | {out["onLoad"]["value"]}
    assert values == {"location"}


def test_a_jump_past_several_sections_lands_on_the_right_one(out):
    """The case that broke the observer version: clicking an entry near the foot.

    The heading crosses the trigger line between two samples, so an observer sees
    "not intersecting" before and after and reports nothing at all.
    """
    assert out["afterJump"]["id"] == out["ids"][-1]


def test_scrolling_back_to_the_top_restores_the_first_entry(out):
    assert out["backToTop"]["id"] == out["ids"][0]


def test_the_card_tracks_resize_as_well_as_scroll(out):
    """Section offsets move when the column reflows, not only when the page scrolls."""
    assert set(out["listens"]) == {"scroll", "resize", "hashchange"}
