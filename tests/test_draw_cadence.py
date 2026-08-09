"""When a drawn stroke is classified, and by what rule.

``digit_source.js`` coalesces pointer moves to one emission per animation frame,
which is enough while a consumer classifies inside a frame budget. The comparison
board does not: two models at ~17 ms and ~129 ms block the main thread for
~146 ms per frame, and because the pad's ink cannot be *painted* until the thread
yields, the stroke arrives in 146 ms jumps. The pad, not just the answer, becomes
unusable.

So an expensive board holds emissions until the pen pauses. Three things have to
stay true for that to work, and all three are the kind that break silently:

* pointer moves must go through the settle path, not straight to ``requestEmit``;
* lifting the pen must pre-empt the timer, or every stroke ends in a needless
  wait;
* an emission carrying an unchanged digit must be dropped, or the duplicate that
  ``pointerup`` produces costs a second full pass for an identical picture.

``mount()`` needs a canvas and a 2D context, and there is no jsdom here, so this
reads the source -- the same approach ``test_stage_projection.py`` takes to the
devicePixelRatio transform, and for the same reason. The behaviour itself is
checked in a browser.
"""
import os
import re

import pytest

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "apps", "web")
SOURCE_JS = os.path.join(WEB, "digit_source.js")
COMPARE_JS = os.path.join(WEB, "d2nn_compare.js")


def read(path):
    return open(path, encoding="utf-8").read()


def strip_comments(src):
    """Prose about a trap must not count as a call site."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", src, flags=re.S)


def handler_body(src, event):
    """The body of a pad.addEventListener("<event>", ...) handler."""
    m = re.search(r'pad\.addEventListener\("%s",\s*\(ev\)\s*=>\s*\{(.*?)\n    \}\);' % event,
                  src, re.S)
    assert m, f"no {event} handler found on the pad -- update this guard if it moved"
    return m.group(1)


# ------------------------------------------------------------------- cadence

@pytest.mark.parametrize("event", ["pointerdown", "pointermove"])
def test_drawing_goes_through_the_settle_path(event):
    """A stroke must not call requestEmit directly.

    This is the regression that would restore per-frame classification while
    drawing, and it would look like nothing at all in a diff -- one identifier.
    """
    body = strip_comments(handler_body(read(SOURCE_JS), event))
    assert "inputChanged()" in body, f"{event} no longer routes through inputChanged"
    assert "requestEmit()" not in body, (
        f"{event} calls requestEmit directly, so an expensive consumer is back to "
        "classifying on every frame of a stroke"
    )


def test_lifting_the_pen_preempts_the_wait():
    """A lift is the clearest pause there is; it should not cost another delay."""
    src = strip_comments(read(SOURCE_JS))
    m = re.search(r"const endStroke = \(\) => \{(.*?)\};", src, re.S)
    assert m, "endStroke not found -- update this guard if it was renamed"
    assert "cancelSettle()" in m.group(1), "ending a stroke leaves the settle timer running"
    assert "requestEmit()" in m.group(1), "ending a stroke no longer emits"


def test_an_unchanged_digit_is_not_re_emitted():
    """The duplicate-suppression that makes the end of a stroke cost one pass."""
    src = strip_comments(read(SOURCE_JS))
    m = re.search(r"function emit\(\) \{(.*?)\n    \}", src, re.S)
    assert m, "emit() not found"
    body = m.group(1)
    assert "unchanged" in body and "if (unchanged) return;" in body, (
        "emit() no longer drops an emission carrying the same digit"
    )
    # `last` must still be updated, or `current()` goes stale for late subscribers.
    assert body.index("last = ") < body.index("if (unchanged) return;"), (
        "emit() returns before recording the digit, so current() would go stale"
    )


def test_the_source_exposes_what_a_board_needs_to_drive_it():
    src = read(SOURCE_JS)
    m = re.search(r"return \{(.*?)\n    \};", src, re.S)
    assert m, "the mount handle was not found"
    for key in ("subscribe", "onPending", "setSettle", "current", "pending"):
        assert key in m.group(1), f"the digit source handle no longer exposes {key}"


# ---------------------------------------------------------------- thresholds

def constants(src, names):
    out = {}
    for n in names:
        m = re.search(rf"const {n} = ([\d.]+);", src)
        assert m, f"{n} not found in d2nn_compare.js"
        out[n] = float(m.group(1))
    return out


def test_the_cadence_thresholds_straddle_a_frame():
    """Enter settle mode above a frame's work, leave below it, never both at once."""
    c = constants(read(COMPARE_JS),
                  ["FRAME_MS", "SETTLE_ENTER_MS", "SETTLE_LEAVE_MS", "SETTLE_MS"])
    assert c["SETTLE_LEAVE_MS"] < c["FRAME_MS"] < c["SETTLE_ENTER_MS"], (
        "the hysteresis band no longer contains one animation frame, so a board "
        f"could flip mode between strokes: {c}"
    )


def test_the_settle_delay_is_a_pause_not_a_wait():
    """Long enough to swallow a stroke, short enough not to read as waiting.

    Moves within a stroke are milliseconds apart, so anything above ~100 ms
    swallows a whole one; much beyond ~500 ms and the board reads as broken
    rather than deliberate.
    """
    c = constants(read(COMPARE_JS), ["SETTLE_MS"])
    assert 100 <= c["SETTLE_MS"] <= 500, f"SETTLE_MS = {c['SETTLE_MS']} ms"


def test_the_cadence_decision_is_measured_not_guessed_from_depth():
    """The board must decide from its own timing, not from a mask count.

    A threshold on n_layers would be wrong on a machine unlike this one, and the
    whole point of measuring is that the same board is fast on a desktop and slow
    on a laptop.
    """
    src = strip_comments(read(COMPARE_JS))
    m = re.search(r"function updateCadence\(elapsed\) \{(.*?)\n    \}", src, re.S)
    assert m, "updateCadence not found"
    body = m.group(1)
    assert "cost" in body and "setSettle" in body
    assert "n_layers" not in body, "the cadence is being decided from model depth, not timing"
    assert "performance.now()" in src, "nothing is being timed"
