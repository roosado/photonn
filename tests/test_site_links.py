"""Every link between pages has to land somewhere.

The site used to be three pages wired together by two hardcoded href tokens, and
eyeballing them was enough. It is now five pages with a topbar listing all of them
and a sequential "next" hand-off at the foot of each, so every page carries six or
more internal links and a rename silently breaks a dozen of them at once.

These pages are also served from ``file://`` as often as from a web server -- the
whole point of inlining everything -- and a broken relative link there fails
silently rather than with a 404 anyone would notice.

``apps/build_site.py`` generates all of it from :data:`apps.build_site.PAGES`, so
what these tests really guard is that the generated set of files and the generated
set of links stay the same set.
"""
import os
import re

import pytest

from apps.build_site import PAGES, href

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "..", "site")

#: Deleted by the 2026-08-10 redesign: its reader-facing half became the front
#: page and its methodology moved to physics.html. Nothing may link to it again.
DELETED_PAGES = ("classifier.html",)

HREF = re.compile(r'href="([^"]+)"')


def page_text(name):
    path = os.path.join(SITE, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not built; run python -m apps.build_site")
    return open(path, encoding="utf-8").read()


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.key)
def test_every_page_in_the_nav_exists(page):
    assert os.path.exists(os.path.join(SITE, page.file)), (
        f"{page.file} is listed in PAGES and linked from every topbar, but was not built"
    )


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.key)
def test_every_internal_link_resolves(page):
    html = page_text(page.file)
    targets = {h for h in HREF.findall(html) if not h.startswith(("http://", "https://", "#", "mailto:"))}
    assert targets, f"{page.file} has no internal links at all; the topbar is missing"
    for target in targets:
        # "./" is the site root, which is index.html once served.
        resolved = "index.html" if target == "./" else target
        assert os.path.exists(os.path.join(SITE, resolved)), (
            f"{page.file} links to {target!r}, which is not a file in site/"
        )


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.key)
def test_nothing_links_to_a_deleted_page(page):
    html = page_text(page.file)
    for gone in DELETED_PAGES:
        assert gone not in html, f"{page.file} still references the deleted {gone}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.key)
def test_every_page_links_to_every_other(page):
    """The topbar is the navigation; a page that dropped out of it is unreachable."""
    html = page_text(page.file)
    for other in PAGES:
        assert f'href="{href(other.key)}"' in html, (
            f"{page.file} does not link to {other.file}; the topbar lost an entry"
        )


#: The attribute as it appears on an anchor. Matching the bare attribute would
#: also hit the CSS rule that styles it.
CURRENT = re.compile(r'<a [^>]*aria-current="page"[^>]*>')


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.key)
def test_exactly_one_nav_entry_is_marked_current(page):
    html = page_text(page.file)
    marked = CURRENT.findall(html)
    assert len(marked) == 1, (
        f"{page.file} must mark exactly one topbar entry as the live page, found {len(marked)}"
    )
    assert href(page.key) in marked[0], (
        f"{page.file} marks the wrong topbar entry as current: {marked[0]}"
    )


@pytest.mark.parametrize("name", [p.file for p in PAGES])
def test_no_page_uses_an_em_dash(name):
    """Site prose convention, set 2026-08-11: no em-dashes anywhere.

    En-dashes stay, in numeric ranges and in compound names like Mach-Zehnder;
    only the em-dash is out. It was enforced by hand until now, and it drifted:
    four pages held at zero while /tolerance accumulated 22, because nothing
    checked. A convention nothing checks is a preference.

    Both spellings are caught. The entity is what the page sources write, and the
    literal character is what a careless paste introduces.
    """
    text = page_text(name)
    assert "&mdash;" not in text, f"{name}: &mdash; entity in prose"
    assert "\u2014" not in text, f"{name}: literal em-dash in prose"


def test_the_artifact_body_uses_absolute_links():
    """The Artifact is a lone body on claude.ai with no sibling files.

    A relative href there resolves against claude.ai and 404s, so this variant
    must carry deployed URLs -- and must not carry a document shell, since the
    host supplies its own.
    """
    html = page_text("_artifact_body.html")
    for page in PAGES:
        assert f'href="{href(page.key, absolute=True)}"' in html, (
            f"the artifact body does not link to {page.file} absolutely"
        )
    assert 'href="./"' not in html and "<!doctype" not in html.lower()
