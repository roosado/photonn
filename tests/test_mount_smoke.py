"""Every widget that can be mounted under Node is mounted, once.

The two widgets with real Node runners (``errors.js``, ``interfere.js``) were the
only ones any test executed. The rest were checked by grepping their own source,
on the stated grounds that "mount() needs a canvas and a 2D context, and there is
no jsdom here" -- true when written, and no longer true once ``dom_stub.js``
existed.

This does not test behaviour. It asserts that a widget still *loads and mounts*:
module scope evaluates, CSS injects, the DOM assembles, the first paint runs. That
is precisely the class of breakage a refactor of the shared canvas code would
cause, and before this nothing could see it.

All ten mount. Getting there needed three things, and each was a real gap rather
than a concession: ``innerHTML`` had to actually parse (six widgets build controls
as a markup string and query the pieces back out), the host div had to be *in* the
document (``explorer.js`` finds its own spans with ``getElementById``), and the
dependency bundles had to load in page order (``d2nn.js`` reads
``window.D2NN_WEIGHTS`` at module scope, so weights first -- the same ordering
hazard ``build_site`` guards for ``mesh_weights.js``).
"""
import json
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "mount_smoke_runner.js")

node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node not on PATH")

#: Every widget in apps/web that mounts into a container. Listed rather than
#: discovered so that a new widget has to be added here deliberately -- the
#: alternative is a guard that silently covers whatever happens to exist.
MOUNTABLE = (
    "errors.js", "interfere.js", "explorer.js", "digit_source.js",
    "d2nn_stage.js", "scaling.js", "optics.js", "analogy.js",
    "d2nn_demo.js", "d2nn_compare.js", "activation.js",
)


@pytest.fixture(scope="module")
def report():
    proc = subprocess.run([node, RUNNER], capture_output=True, text=True)
    assert proc.returncode == 0, f"mount smoke runner failed:\n{proc.stderr}"
    return {r["widget"]: r for r in json.loads(proc.stdout)["results"]}


@pytest.mark.parametrize("widget", MOUNTABLE)
def test_widget_mounts_without_throwing(widget, report):
    r = report[widget]
    assert r["status"] == "ok", f"{widget}: {r['status']} -- {r['detail']}"


@pytest.mark.parametrize("widget", MOUNTABLE)
def test_widget_reaches_for_no_undeclared_canvas_method(widget, report):
    """A method the stub never declared is a call nobody anticipated.

    Following ``dom_stub``'s convention: a widget reaching past what its harness
    declares should surface, not silently no-op.
    """
    assert report[widget]["unknownCtxCalls"] == [], (
        f"{widget} called canvas methods the stub does not model: "
        f"{report[widget]['unknownCtxCalls']}"
    )


def test_every_widget_is_accounted_for(report):
    """The runner and this list must not drift apart."""
    assert set(report) == set(MOUNTABLE)


def test_each_widget_actually_builds_dom(report):
    """A mount that throws nothing but renders nothing is not a mounted widget."""
    empty = [w for w in MOUNTABLE if report[w]["status"] == "empty"]
    assert not empty, f"mounted but produced no DOM: {empty}"
