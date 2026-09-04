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


def style_id_owners():
    """Every file in apps/web that declares one, discovered rather than listed."""
    return sorted({f for files in style_ids().values() for f in files})


def test_at_least_the_known_widgets_declare_one():
    """Guard the guard: a renamed constant would make this file assert nothing."""
    owners = style_id_owners()
    for expected in ("d2nn_stage.js", "digit_source.js", "d2nn_compare.js", "interfere.js"):
        assert expected in owners, f"{expected} no longer declares a STYLE_ID"


def test_no_two_widgets_share_a_style_id():
    clashes = {sid: files for sid, files in style_ids().items() if len(files) > 1}
    assert not clashes, (
        "these widgets share a <style> id, so whichever mounts second on a page "
        f"will silently run with no CSS: {clashes}"
    )


@pytest.mark.parametrize("widget", style_id_owners())
def test_the_injected_css_is_namespaced_to_the_id_owner(widget):
    """A widget's rules should live under its own root class.

    Distinct ids stop the *injection* from being skipped; distinct class prefixes
    stop the rules that do get injected from reaching into another widget.

    Parametrized over whatever declares a STYLE_ID rather than over a list kept by
    hand: this is a property every such widget has to hold, and a hand-kept list
    silently exempts the next one somebody adds.
    """
    prefixes = css_prefixes(widget)
    assert len(prefixes) <= 2, (
        f"{widget} styles several unrelated class prefixes {sorted(prefixes)}; "
        "its rules can reach into other widgets"
    )


def css_prefixes(widget):
    """The class prefixes a widget's stylesheet writes rules for."""
    src = open(os.path.join(WEB, widget), encoding="utf-8").read()
    css = re.search(r"const CSS = `(.*?)`", src, re.S)
    assert css, f"{widget} has no CSS template literal"
    selectors = re.findall(r"^\s*([.#][\w-]+)", css.group(1), re.M)
    assert selectors, f"{widget} CSS has no top-level selectors"
    return {s.split("-")[0].lstrip(".#") for s in selectors}


def test_no_two_widgets_share_a_class_prefix():
    """The other half of the same rule, and the half that stayed broken.

    A unique id stops the *injection* being skipped. It does nothing about two
    widgets writing rules for the same classes: both stylesheets then land, every
    rule applies to both widgets, and the one that injected last wins.

    That is exactly what survived the ``ds-style`` fix. The id was renamed and the
    class prefix was not, so ``d2nn_stage.js`` and ``digit_source.js`` went on
    sharing ``.ds-root``, ``.ds-seg`` and ``.ds-btn`` -- and on the optics page,
    where both mount, the source's segmented control silently rendered at the
    stage's padding and both roots took the stage's gap.

    The test above asks whether one widget uses too many prefixes. Neither file
    ever failed it. This asks the question that was actually being got wrong.
    """
    owners = defaultdict(list)
    for widget in style_id_owners():
        for prefix in css_prefixes(widget):
            owners[prefix].append(widget)
    clashes = {p: files for p, files in owners.items() if len(files) > 1}
    assert not clashes, (
        "these widgets write rules for the same class prefix, so on a page "
        "carrying both, the one that injects last restyles the other: "
        f"{ {p: sorted(f) for p, f in clashes.items()} }"
    )
