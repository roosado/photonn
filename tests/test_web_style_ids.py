"""Every browser widget must own a distinct <style> element id.

Each widget in ``apps/web`` injects its CSS once, guarded by

    if (document.getElementById(STYLE_ID)) return;

which is the right thing to do for a widget mounted more than once on a page --
and quietly the wrong thing when two *different* widgets pick the same id. The
second one to mount then finds the id taken, returns, and runs with none of its
own CSS.

That is what ``d2nn_stage.js`` and ``digit_source.js`` both claiming ``ds-style``
did. It was invisible while no page carried both. Once the optics page carried
the comparison board (which mounts digit_source) and the 3D stage together, the
stage lost: its flex toolbar fell back to block layout and its canvas collapsed
to the 300x150 default, on a page that had passed every test.

Nothing at runtime complains about this, and it cannot be caught by rendering one
widget at a time, so it is asserted over the source instead.
"""
import os
import re
from collections import defaultdict

import pytest

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "apps", "web")

STYLE_ID_RE = re.compile(r'const\s+STYLE_ID\s*=\s*"([^"]+)"')


def style_ids():
    """{style id -> [files that claim it]} across every widget in apps/web."""
    claims = defaultdict(list)
    for name in sorted(os.listdir(WEB)):
        if not name.endswith(".js"):
            continue
        src = open(os.path.join(WEB, name), encoding="utf-8").read()
        for sid in STYLE_ID_RE.findall(src):
            claims[sid].append(name)
    return claims


def test_at_least_the_known_widgets_declare_one():
    """Guard the guard: a renamed constant would make this file assert nothing."""
    claims = style_ids()
    owners = {f for files in claims.values() for f in files}
    for expected in ("d2nn_stage.js", "digit_source.js", "d2nn_compare.js", "interfere.js"):
        assert expected in owners, f"{expected} no longer declares a STYLE_ID"


def test_no_two_widgets_share_a_style_id():
    clashes = {sid: files for sid, files in style_ids().items() if len(files) > 1}
    assert not clashes, (
        "these widgets share a <style> id, so whichever mounts second on a page "
        f"will silently run with no CSS: {clashes}"
    )


@pytest.mark.parametrize("widget", ["d2nn_stage.js", "digit_source.js", "d2nn_compare.js",
                                   "interfere.js"])
def test_the_injected_css_is_namespaced_to_the_id_owner(widget):
    """A widget's rules should live under its own root class.

    Distinct ids stop the *injection* from being skipped; distinct class prefixes
    stop the rules that do get injected from reaching into another widget.
    """
    src = open(os.path.join(WEB, widget), encoding="utf-8").read()
    css = re.search(r"const CSS = `(.*?)`", src, re.S)
    assert css, f"{widget} has no CSS template literal"
    selectors = re.findall(r"^\s*([.#][\w-]+)", css.group(1), re.M)
    assert selectors, f"{widget} CSS has no top-level selectors"
    prefixes = {s.split("-")[0].lstrip(".#") for s in selectors}
    assert len(prefixes) <= 2, (
        f"{widget} styles several unrelated class prefixes {sorted(prefixes)}; "
        "its rules can reach into other widgets"
    )
