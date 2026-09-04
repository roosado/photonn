"""A widget that paints from the page's palette must hear the theme change.

The site's theme toggle sets ``data-theme`` on ``<html>`` and persists it; it
never touches the OS preference. So a canvas widget that reads its ink out of the
CSS custom properties and listens only on
``matchMedia("(prefers-color-scheme:dark)")`` hears nothing when a reader flips
the toggle: the card around the chart restyles instantly, because that is CSS,
while the chart inside it keeps the previous theme's ink until something else
happens to trigger a redraw.

Three widgets had that gap at once (``scaling.js`` and ``optics.js`` on /optics,
``explorer.js`` on /physics) while two others did it correctly, because "how a
widget learns the theme changed" was a convention each file re-invented rather
than a rule anything checked.

It is one rule now: ``plot.js`` owns both sources, and a widget calls
``P.onThemeChange``. So the checks are (a) the shared module watches the right
attribute, and (b) no widget has grown a private observer again.

None of this is catchable in the browser we drive -- that tab is always hidden,
so it never paints. Asserted over the source.
"""
import os
import re

import pytest

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "apps", "web")

PLOT = "plot.js"
#: Observes the attribute the toggle actually writes.
WATCHES_THEME = re.compile(r'attributeFilter\s*:\s*\[\s*"data-theme"\s*\]')
#: Reads an ink colour out of the page's custom properties.
READS_PALETTE = re.compile(r"getComputedStyle\s*\(")
#: Delegates to the shared module.
DELEGATES = re.compile(r"\bP\.onThemeChange\s*\(")
#: Uses the shared palette.
USES_PALETTE = re.compile(r"\bP\.palette\b|\bP\.readVars\b")


def widgets():
    return sorted(n for n in os.listdir(WEB) if n.endswith(".js"))


def source(name):
    with open(os.path.join(WEB, name), encoding="utf-8") as fh:
        return fh.read()


def test_the_shared_module_watches_the_attribute_the_toggle_writes():
    src = source(PLOT)
    assert WATCHES_THEME.search(src), (
        "plot.js must observe data-theme on the document element; matchMedia "
        "alone never fires for this site's own toggle"
    )
    assert "matchMedia" in src, (
        "and matchMedia too, for a reader on 'system' who never touches the toggle"
    )


def test_only_the_shared_module_reads_the_palette():
    """One place converts CSS custom properties into ink."""
    readers = [n for n in widgets() if READS_PALETTE.search(source(n))]
    assert readers == [PLOT], (
        f"{[n for n in readers if n != PLOT]} read the palette directly instead of "
        "calling P.palette, so the site's ink is defined in more than one place"
    )


def test_no_widget_has_grown_a_private_theme_observer():
    """The regression this file exists to prevent, in its general form."""
    private = [n for n in widgets() if n != PLOT and WATCHES_THEME.search(source(n))]
    assert not private, (
        f"{private} observe data-theme themselves rather than calling "
        "P.onThemeChange. That is how the two conventions -- and the wrong one -- "
        "arose in the first place."
    )


@pytest.mark.parametrize("widget", [
    "scaling.js", "optics.js", "explorer.js", "d2nn_stage.js", "analogy.js",
])
def test_a_palette_user_repaints_on_a_theme_change(widget):
    """Guard the guard: these five paint theme-dependent pixels and must react."""
    src = source(widget)
    assert USES_PALETTE.search(src), f"{widget} no longer uses the shared palette"
    assert DELEGATES.search(src), (
        f"{widget} paints from the page palette but never calls P.onThemeChange, "
        "so the theme toggle leaves its canvas in the previous theme's ink"
    )
