"""The in-page contents card, and the section ids it points at.

The card is *generated from the markup* (``apps.build_site.section_index``) rather
than authored beside it, on the same bargain ``PAGES`` makes for the topbar: adding
a section to a page adds it to that page's index, with nothing to remember. Four
things have to hold for that to be true and to stay true:

* **Every fragment link resolves inside its own page.** ``test_site_links.py``
  deliberately exempts ``#`` hrefs from its link check (they are not files), so
  without this nothing would catch a contents entry pointing at an id that a prose
  edit renamed.
* **The card's active entry is never ``aria-current="page"``.**
  ``test_site_links.py`` asserts exactly one such anchor per page and that it is the
  topbar's own. ``location`` is both the correct ARIA value for an in-page index and
  what keeps that assertion meaningful rather than accidentally satisfied.
* **The nesting is the page's own outline.** A section written as ``<h3>`` belongs to
  the band above it; one written as ``<h2>`` does not. If the card derived nesting
  from sibling order instead, ``/tolerance``'s closing chip section -- which follows
  two bands but belongs to neither -- would be silently indented under one.
* **/tolerance stays grouped.** Issue #6 adds four geometry sources to the Setup
  band; this pins the structure they land in.

Reads the built pages, so it skips cleanly when ``site/`` has not been generated.
"""
import os
import re

import pytest

from apps.build_site import PAGES, section_index, slugify, toc, toc_label

SITE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "site")

#: The card is one <nav>; everything below is parsed out of it.
TOC_BLOCK = re.compile(r'<nav class="toc[^"]*" aria-label="On this page">.*?</nav>', re.S)
TOC_ITEM = re.compile(r'<li([^>]*)><a href="#([^"]+)">(.*?)</a></li>', re.S)
IDS = re.compile(r'\bid="([^"]+)"')
FRAGMENTS = re.compile(r'href="#([^"]+)"')


def page_text(name):
    path = os.path.join(SITE, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not built")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def toc_of(text):
    block = TOC_BLOCK.search(text)
    assert block, "page carries no contents card"
    return TOC_ITEM.findall(block.group(0))


ALL_PAGES = [p.file for p in PAGES]


# ------------------------------------------------------------------ the anchors

@pytest.mark.parametrize("name", ALL_PAGES)
def test_every_fragment_link_resolves_within_its_page(name):
    text = page_text(name)
    ids = set(IDS.findall(text))
    missing = sorted({f for f in FRAGMENTS.findall(text) if f not in ids})
    assert not missing, f"{name}: fragment links with no target: {missing}"


@pytest.mark.parametrize("name", ALL_PAGES)
def test_section_ids_are_unique(name):
    found = IDS.findall(page_text(name))
    dupes = sorted({i for i in found if found.count(i) > 1})
    assert not dupes, f"{name}: duplicate ids {dupes}"


@pytest.mark.parametrize("name", ALL_PAGES)
def test_every_section_heading_is_addressable(name):
    """A section heading with no id is a section the card cannot reach.

    Two kinds of heading are deliberately not sections and carry no id: the footer's
    three column labels, and ``.sub-h``, which marks a turn *within* a section's
    prose. Indexing those would put entries in the card that do not correspond to
    anything a reader would call a section.
    """
    text = page_text(name)
    headings = re.findall(r"<(h2|h3)([^>]*)>", text)
    bare = [f"<{tag}{attrs}>" for tag, attrs in headings
            if 'id="' not in attrs and "sub-h" not in attrs]
    footer = text[text.index("<footer"):] if "<footer" in text else ""
    bare = [h for h in bare if h not in footer]
    assert not bare, f"{name}: headings with no id: {bare}"


# --------------------------------------------------------------------- the card

@pytest.mark.parametrize("name", ALL_PAGES)
def test_the_card_has_an_entry_for_every_section(name):
    text = page_text(name)
    entries = toc_of(text)
    body = text[:text.index("<footer")] if "<footer" in text else text
    sections = re.findall(r"<(?:h2|h3)[^>]*\bid=", body)
    assert len(entries) == len(sections), f"{name}: {len(entries)} entries, {len(sections)} sections"


@pytest.mark.parametrize("name", ALL_PAGES)
def test_the_card_never_claims_to_be_the_current_page(name):
    """`aria-current="page"` belongs to the topbar; the card marks `location`."""
    block = TOC_BLOCK.search(page_text(name))
    assert block and 'aria-current="page"' not in block.group(0)


@pytest.mark.parametrize("name", ALL_PAGES)
def test_card_order_matches_document_order(name):
    text = page_text(name)
    positions = [text.index(f'id="{ident}"') for _, ident, _ in toc_of(text)]
    assert positions == sorted(positions), f"{name}: card is out of document order"


# ------------------------------------------------------------- /tolerance groups

def test_tolerance_groups_its_sources_into_two_families():
    entries = toc_of(page_text("tolerance.html"))
    bands = [ident for cls, ident, _ in entries if "toc-g" in cls]
    assert bands == ["fabrication", "setup"]


def test_each_family_holds_its_own_numbered_sources():
    """Six device sources under Fabrication, four geometry ones under Setup.

    The numbers come from the eyebrows ("Source N of M"), so they restart at 1 in
    each family rather than running 1-10 across the page. That is what makes the
    families readable, and it is only correct as long as every numbered source
    actually sits under the band whose count it is quoting.
    """
    entries = toc_of(page_text("tolerance.html"))
    idx = {ident: i for i, (_, ident, _) in enumerate(entries)}
    numbers = {ident: int(re.search(r'toc-n">(\d+)<', label).group(1))
               for _, ident, label in entries if "toc-n" in label}

    fabrication = [i for i in numbers if idx["fabrication"] < idx[i] < idx["setup"]]
    setup = [i for i in numbers if idx[i] > idx["setup"]]

    assert len(numbers) == 10, "every source should be numbered"
    assert sorted(numbers[i] for i in fabrication) == [1, 2, 3, 4, 5, 6]
    assert sorted(numbers[i] for i in setup) == [1, 2, 3, 4]


def test_the_chip_comparison_belongs_to_neither_family():
    """It is the cross-machine comparison, not a seventh error source.

    It follows both bands, so a card that nested by sibling order would indent it
    under Setup. It is an <h2>, so a card that nests by outline does not.
    """
    entries = toc_of(page_text("tolerance.html"))
    cls, ident, _ = entries[-1]
    assert ident == "the-interferometer-chip"
    assert "toc-s" not in cls and "toc-g" not in cls


def test_the_chip_section_is_named_not_alluded_to():
    text = page_text("tolerance.html")
    assert "The other machine fails a different way" not in text
    assert "The interferometer chip fails a different way" in text


# ------------------------------------------------------------------- unit-level

@pytest.mark.parametrize("text,want", [
    ("Crosstalk: pixels will not stay out of each other", "Crosstalk"),
    ("Why they are the same machine", "Why they are the same machine"),
    ("Made of <em>light</em>", "Made of light"),
    ("The deep model&rsquo;s budget", "The deep model’s budget"),
])
def test_toc_label_shortens_only_where_there_is_a_topic_to_name(text, want):
    assert toc_label(text) == want


def test_slugify_survives_markup_entities_and_punctuation():
    assert slugify("Why build a computer out of <em>light</em>?") == "why-build-a-computer-out-of-light"
    assert slugify("&mdash; &mdash;") == "section"


def test_repeated_headings_get_distinct_ids():
    body = (
        '<p class="eyebrow">Source 1 of 6</p>\n      <h2>Loss: one way</h2>'
        '<p class="eyebrow">Source 2 of 6</p>\n      <h2>Loss: another</h2>'
    )
    html, entries = section_index(body)
    assert [e["id"] for e in entries] == ["loss", "loss-2"]
    assert [e["num"] for e in entries] == ["1", "2"]
    assert 'id="loss"' in html and 'id="loss-2"' in html


def test_a_page_with_no_sections_gets_no_card():
    assert toc([]) == ""
