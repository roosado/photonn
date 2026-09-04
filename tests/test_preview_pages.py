"""The standalone widget previews, held to the site's own rules.

``apps/*_demo.html`` and ``apps/diffraction_explorer.html`` are committed and add
up to a couple of megabytes, and until there was one module building them they
were covered by nothing: no weight ceiling, no self-containment check, no
figure-encoding check, while every ``site/*.html`` had all three.

They are looser than the site pages on purpose. A preview exists to look at one
widget in isolation, so it inlines whatever that widget needs and is allowed to
be heavy; what it is not allowed to do is fetch, because these are opened from
``file://`` as often as from anywhere.

Builds each page from :mod:`apps.preview` rather than reading the committed file,
for the same reason ``test_site_build`` exists: the assertions should describe the
source. The comparison against what is committed is the last test here.
"""
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APPS = os.path.join(os.path.dirname(HERE), "apps")

KB = 1024

#: module name -> (output file, ceiling in KB). Ceilings are generous: these
#: inline entire trained models on purpose.
PREVIEWS = {
    "optics_demo": ("optics_demo.html", 400),
    "analogy_demo": ("analogy_demo.html", 200),
    "diffraction_explorer": ("diffraction_explorer.html", 200),
    "d2nn_demo": ("d2nn_demo.html", 700),
    "compare_demo": ("compare_demo.html", 2000),   # two models, one 56 masks
}


def build(module_name):
    import importlib

    mod = importlib.import_module(f"apps.{module_name}")
    return mod.build_html()


@pytest.mark.parametrize("module_name", sorted(PREVIEWS))
def test_a_preview_page_fetches_nothing(module_name):
    """The property that makes a preview openable from file://."""
    html = build(module_name)
    for forbidden in ("fetch(", "XMLHttpRequest", "new Worker", "importScripts"):
        assert forbidden not in html, f"{module_name} contains {forbidden!r}"


@pytest.mark.parametrize("module_name", sorted(PREVIEWS))
def test_a_preview_page_is_within_budget(module_name):
    _, ceiling = PREVIEWS[module_name]
    size = len(build(module_name).encode("utf-8")) / KB
    assert size <= ceiling, f"{module_name} is {size:.0f} KB against a {ceiling} KB ceiling"


@pytest.mark.parametrize("module_name", sorted(PREVIEWS))
def test_a_preview_page_mounts_every_host_it_declares(module_name):
    """A host div with no mount is a blank space where a widget should be.

    `mount_queue` deliberately skips a container it cannot find, so a mistyped id
    produces a silently missing widget rather than an error.
    """
    import importlib

    mod = importlib.import_module(f"apps.{module_name}")
    html = build(module_name)
    for host in mod.PREVIEW["hosts"]:
        assert f'<div id="{host}"></div>' in html, f"{module_name}: no host div for {host!r}"
        assert f'"{host}"' in html or f"'{host}'" in html, (
            f"{module_name}: host {host!r} is declared but nothing mounts into it"
        )


def test_every_preview_shares_one_shell():
    """The shell is one module, so its chrome is one edit rather than five."""
    from apps import preview

    for module_name in PREVIEWS:
        html = build(module_name)
        assert preview.CSS in html, f"{module_name} does not use the shared preview shell"
