"""The committed site is the render of the current source.

``site/*.html`` is committed and CI uploads it verbatim, so the built page is the
deliverable rather than a build artifact. That makes it drift-prone in a way a
generated-at-deploy-time site is not: the only thing keeping ``site/`` level with
``apps/build_site.py`` was remembering to run the build.

This is the test that notices. It is also why ``main()`` writes with
``newline="\\n"`` -- without it Python translates on Windows and the comparison
below is against bytes no one intended to commit.
"""
import os

import pytest

from apps.build_site import PAGES
from built_site import page_html

SITE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "site")

#: Every file `render()` emits: one per page, plus the standalone Artifact body.
FILES = tuple(p.file for p in PAGES) + ("_artifact_body.html",)


@pytest.mark.parametrize("name", FILES)
def test_the_committed_page_is_the_render_of_the_source(name):
    path = os.path.join(SITE, name)
    assert os.path.exists(path), (
        f"site/{name} has never been built. Run `python -m apps.build_site`."
    )
    # newline="" so the comparison is against the bytes on disk, not against a
    # line-ending translation of them.
    with open(path, encoding="utf-8", newline="") as fh:
        on_disk = fh.read()
    assert on_disk == page_html(name), (
        f"site/{name} is not what apps.build_site.render() produces now. "
        "Run `python -m apps.build_site` and commit the result."
    )
