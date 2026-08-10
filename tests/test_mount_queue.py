"""The gate that lets widgets start -- and every way it must not jam.

``apps/web/mount_queue.js`` holds every widget on a page until after the first
paint, then starts them one at a time. That makes it the one piece of front-end
code whose failure is total: a gate that never opens leaves the reader looking at
"warming up" forever, with no error and nothing to click. The widgets themselves
would still pass all their own tests, because they are simply never called.

It cannot be checked in the driven browser either -- that tab is always hidden,
so it never paints, never fires requestAnimationFrame and never delivers an
IntersectionObserver callback. Finding that out is *how* the hidden-document case
below came to be written: the first version of the scheduler waited on a paint
signal alone and stranded every widget on a page opened in a background tab.

So it is exercised under Node against a stand-in for the parts of the DOM it
touches, with time, paint and scrolling driven by hand. See
``tests/mount_queue_runner.js``.
"""
import json
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "mount_queue_runner.js")

node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node not on PATH; mount scheduler checks skipped")


@pytest.fixture(scope="module")
def out():
    proc = subprocess.run([node, RUNNER], capture_output=True, text=True)
    assert proc.returncode == 0, f"mount queue runner failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def test_nothing_runs_before_the_first_paint(out):
    """The whole point: the page must be readable before any widget starts."""
    assert out["ordering"]["beforePaint"] == []


def test_every_widget_starts_itself_in_document_order(out):
    """No taps, no buttons, no small-screen fallback -- just sequenced."""
    assert out["ordering"]["final"] == ["a", "b", "c"]
    assert out["ordering"]["pendingClassCleared"], "the 'warming up' class outlived the mount"


def test_a_hidden_document_does_not_wait_for_a_paint_that_never_comes(out):
    """Opening a link in a background tab, or restoring a session, lands here.

    A hidden tab never paints and never fires requestAnimationFrame, so waiting
    for one would queue every widget forever. This is the regression that the
    driven-browser check actually caught.
    """
    assert out["hiddenRunsAnyway"] == ["a"]


def test_a_host_without_requestanimationframe_still_mounts(out):
    assert out["noRafRunsAnyway"] == ["a"]


def test_a_paint_that_never_arrives_cannot_strand_the_page(out):
    """Belt and braces: even a visible document gets a hard deadline."""
    assert out["paintTimeout"]["at500"] == [], "started before the paint deadline"
    assert out["paintTimeout"]["at3000"] == ["a"], "never started despite the deadline"


def test_a_deferred_widget_waits_for_the_reader_then_starts_by_itself(out):
    """Below-the-fold work warms up on approach -- still with nothing to press."""
    assert out["deferred"]["beforeScroll"] == ["near"]
    assert out["deferred"]["afterScroll"] == ["near", "far"]
    assert out["deferred"]["rootMargin"] == "600px", (
        "the margin is the whole safety factor: a widget still blank when it "
        "reaches the viewport has spent the reader's attention to save load time"
    )


def test_a_deferred_widget_without_intersectionobserver_still_runs(out):
    assert out["deferredWithoutIo"] == ["far"]


def test_one_broken_widget_does_not_strand_the_rest(out):
    assert out["survivesThrow"] == ["bad", "good"]


def test_an_absent_container_is_skipped(out):
    """A page that carries only some widgets must not jam on the ones it lacks."""
    assert out["missingIdSkipped"] == ["real"]
