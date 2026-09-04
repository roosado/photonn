"""The site as ``render()`` produces it, built once per session.

Every site test used to read ``site/*.html`` off disk. That is the right thing to
measure -- the committed bytes are what CI uploads, so they are what a visitor
gets -- but on its own it left two holes.

Nothing called ``render()``. Edit a body string in ``apps/build_site.py``, forget
to run the build, commit: the suite stays green and the deployed page is the
previous one. And every module skipped when the file was absent, so on a machine
without a build roughly seventy assertions did not fail, they quietly did not run.

Building once per session and asserting the committed bytes match (see
``test_site_build.py``) closes both: the assertions below describe the current
source, and the one test that compares against disk says so plainly when the two
have drifted.

Ten seconds, once, dominated by encoding the figures five ways each to keep the
smallest -- so it is cached rather than rebuilt per module.
"""
from __future__ import annotations

import functools


@functools.lru_cache(maxsize=None)
def pages() -> dict:
    """``{filename: html}`` for every file the site is made of."""
    from apps.build_site import render

    return render()


def page_html(name: str) -> str:
    """One rendered page by filename."""
    built = pages()
    if name not in built:
        raise AssertionError(
            f"{name} is not one of the files render() produces: {sorted(built)}"
        )
    return built[name]
