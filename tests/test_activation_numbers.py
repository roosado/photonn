"""The activation page may only quote accuracies its written record states.

``site/README.md`` admits that the site's quoted figures are hand-maintained and that
nothing checks them against the model -- and the tolerance edges once sat stale on the
live site for four months because of it. The activation page is new and quotes more
four-place accuracies than any other, so it gets the guard the others never had.

This does not check the page against the model. It checks it against the documents of
record, ``docs/phase5_activation.md`` and ``docs/tolerance_mesh.md``, which
``docs/README.md`` names as the authority: if the two disagree, the documents are
right, and this test makes sure they cannot disagree silently.
"""
import os
import re

from built_site import page_html

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")
RECORD = ("phase5_activation.md", "tolerance_mesh.md")

#: Four places, 0.1000 and up: every accuracy on the page, and not the tolerances, which
#: are quoted as 0.0003-style numbers the record writes as 3 x 10^-4.
ACCURACY = re.compile(r"(?<![\d.])0\.[1-9]\d{3}(?!\d)")


def _record():
    return "\n".join(open(os.path.join(DOCS, f), encoding="utf-8").read() for f in RECORD)


def _prose(html):
    """The page with its inlined scripts, styles and figures removed: prose only."""
    html = re.sub(r"<script>.*?</script>|<style>.*?</style>", "", html, flags=re.S)
    return re.sub(r'src="data:[^"]+"', "", html)


def test_every_accuracy_on_the_page_is_in_the_record():
    page = _prose(page_html("activation.html"))
    record = _record()
    quoted = sorted(set(ACCURACY.findall(page)))
    assert quoted, "the page quotes no four-place accuracy; the guard would assert nothing"
    missing = [q for q in quoted if q not in record]
    assert not missing, (
        f"activation.html quotes {missing}, which docs/phase5_activation.md and "
        "docs/tolerance_mesh.md never state. Put the number in the record first."
    )


def test_the_guard_sees_the_headline_numbers():
    """Guard the guard: the page's own anchors are four-place numbers it must find."""
    page = _prose(page_html("activation.html"))
    for anchor in ("0.8864", "0.8555", "0.8598", "0.8946"):
        assert anchor in page, anchor
