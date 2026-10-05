"""A widget's presence on a page is declared once and checked against the page.

Getting a widget onto a page used to be six coordinated edits: a host ``<div>``
inside a body string, a ``@@X_BUNDLE@@``/``@@X_MOUNT@@`` token pair, a bundle
function, a mount function, and two ``.replace()`` calls in ``render()``. Nothing
checked that a host had a mount or that a mount had a host.

That matters more than it sounds, because ``mount_queue.js`` *deliberately* skips
a container it cannot find -- ``test_mount_queue.py`` asserts that as a feature,
correctly, since a bundle shared across pages must tolerate absent hosts. The
consequence is that a mistyped id produces a page with a silently missing widget
and a green suite.

``Page.widgets`` is the declaration; these are the checks that make it mean
something.
"""
import re

import pytest

from apps.build_site import PAGES
from built_site import page_html

#: A host container as the page bodies write them: an id'd div with no content,
#: which the widget fills in on mount.
HOST = re.compile(r'<div id="([^"]+)"[^>]*></div>')


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.key)
def test_declared_widgets_match_the_hosts_in_the_page(page):
    found = tuple(HOST.findall(page_html(page.file)))
    assert found == tuple(page.widgets), (
        f"{page.file} carries hosts {found} but declares {tuple(page.widgets)}. "
        "A host with no declaration escapes every check here; a declaration with "
        "no host is a widget that will never mount."
    )


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.key)
def test_every_declared_widget_is_mounted(page):
    """A host div with nothing mounting into it is a blank space on the page."""
    html = page_html(page.file)
    for host in page.widgets:
        assert f'"{host}"' in html or f"'{host}'" in html, (
            f"{page.file} declares host {host!r} but no mount script names it"
        )


def test_the_mesh_bundle_precedes_the_widget_that_reads_it():
    """The one script-order rule on the site, asserted on the built page.

    ``errors.js`` reads ``window.PHOTONN_MESH`` at module scope. Emitted after
    the widget, the mesh bundle arrives too late and the widget falls back to a
    synthetic stand-in -- drawing a chip nobody trained, with no error anywhere.
    The rule lived in a docstring and was enforced by a single ``+`` between two
    function calls, which no test could see.

    ``test_mesh_web.py`` cannot catch this: it injects ``PHOTONN_MESH`` itself and
    never reads the built page.
    """
    html = page_html("tolerance.html")
    # Markers unique to each file, not shared strings: errors.js *reads*
    # window.PHOTONN_MESH, so searching for that finds whichever script came
    # first and the check would pass either way round.
    publisher = html.find("window.PHOTONN_MESH = W")   # mesh_weights.js publishes
    reader = html.find("ex-style")                     # errors.js STYLE_ID
    assert publisher >= 0, "the mesh bundle is not on the page at all"
    assert reader >= 0, "errors.js is not on the page at all"
    assert publisher < reader, (
        "mesh_weights.js is emitted after errors.js, so the mesh widget will draw "
        "its synthetic stand-in instead of the trained chip"
    )
    # mesh.js too, since the mesh build moved out of errors.js: it is read at
    # module scope, and a page that emitted it late would throw before mounting
    # any of the seven widgets.
    engine = html.find("window.PhotonnMesh = api")      # mesh.js publishes
    assert engine >= 0, "mesh.js is not on the page at all"
    assert engine < reader, "mesh.js is emitted after errors.js, which reads it at load"
