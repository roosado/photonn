"""Geometry invariants the optics sweep rests on.

Three things in ``apps/sweep_optics.py`` and ``apps/export_sweep_web.py`` became
load-bearing once the sweep could run at more than one grid size, and none of
them is checked by the existing node-gated widget tests:

* **The reach a design requires scales with the grid.** ``default_regions`` and
  ``train._embed`` place everything as fractions of ``n``, so a bigger grid is a
  proportionally bigger device. Reading a 256 wrap budget against the 128
  requirement would overstate the headroom by 2x.
* **Wrap error depends on total reach alone**, not on how reach is split between
  separation and mask count. The entire iso-reach arm is built on this: it is
  what makes "same budget, more masks" a fair comparison rather than a
  confounded one.
* **"Same detector layout" means same *fractions*, not same pixels.** The
  exporter's original guard required identical pixel boxes, which silently
  becomes wrong the moment two models have different grids.

Pure Python and cheap -- no Node, no trained model, no gitignored file.
"""
import numpy as np
import pytest

from apps.export_sweep_web import check_layout
from apps.sweep_optics import DX, WAVELENGTH, config_tag, required_reach_px, stack_wraparound
from photonn.detect import default_regions

#: The published Phase-3 correspondence result (docs/phase3_mesh.md): the worst
#: input pixel sits 74 px from the detector pixel farthest from it, at 128.
SHIPPED_REQUIRED_PX = 74.0


def canonical_regions(grid, **kw):
    return [[r.y0, r.y1, r.x0, r.x1] for r in default_regions(grid, 10, **kw)]


# ------------------------------------------------------------------ requirement

def test_required_reach_reproduces_the_published_figure():
    assert required_reach_px(128) == SHIPPED_REQUIRED_PX


@pytest.mark.parametrize("grid", [128, 256, 512])
def test_required_reach_scales_with_the_grid(grid):
    """Exactly linear in n -- once the inclusive index is accounted for.

    This is the fact that stops a bigger grid being free headroom: the wrap
    budget grows, but so does what the budget has to cover.

    The requirement is a distance between *inclusive* pixel indices, so it is a
    span minus one: 74 px at 128 is really 75 pixels of extent. That trailing
    ``-1`` does not scale, which is why the raw numbers (74, 149, 299) look 1-3 px
    short of doubling while ``need + 1`` (75, 150, 300) doubles exactly.
    """
    assert required_reach_px(grid) + 1 == (SHIPPED_REQUIRED_PX + 1) * grid / 128


# ------------------------------------------------------------------------ wrap

def test_wrap_depends_on_total_reach_not_on_how_it_is_split():
    """Two configs with equal total reach must wrap equally, at any z/L split.

    The iso-reach arm compares mask counts at a fixed reach budget and reads any
    accuracy difference as depth. That is only legitimate if the deeper config is
    not also being handed a cleaner simulation -- which is what this pins.
    """
    grid = 64
    rng = np.random.default_rng(20260807)
    win = grid // 2
    off = (grid - win) // 2
    fields = np.zeros((2, grid, grid), dtype=complex)
    fields[:, off:off + win, off:off + win] = rng.random((2, win, win))

    # z * (L + 1) held constant: 1 mask over 2 hops, 3 masks over 4 hops.
    a = stack_wraparound(4e-3, 2, fields, grid=grid)
    b = stack_wraparound(2e-3, 4, fields, grid=grid)

    assert a["logit_error"] == pytest.approx(b["logit_error"], rel=1e-6)
    assert a["plane_error"] == pytest.approx(b["plane_error"], rel=1e-6)


def test_wrap_grows_with_total_reach():
    """The measurement is not simply insensitive -- more reach must wrap more."""
    grid = 64
    fields = np.zeros((1, grid, grid), dtype=complex)
    fields[:, 16:48, 16:48] = 1.0
    near = stack_wraparound(1e-3, 2, fields, grid=grid)
    far = stack_wraparound(8e-3, 2, fields, grid=grid)
    assert far["plane_error"] > near["plane_error"]


# ---------------------------------------------------------------------- layout

def test_layout_guard_accepts_the_same_design_at_a_larger_grid():
    """A 256 model is the same layout at twice the pixel coordinates."""
    shipped = {"n": 128, "regions": canonical_regions(128)}
    check_layout(canonical_regions(256), 256, shipped)          # must not raise


def test_layout_guard_accepts_an_identical_grid():
    shipped = {"n": 128, "regions": canonical_regions(128)}
    check_layout(canonical_regions(128), 128, shipped)          # must not raise


def test_layout_guard_rejects_a_genuinely_different_layout():
    """A changed field_frac is a real mismatch and must still be caught."""
    shipped = {"n": 128, "regions": canonical_regions(128)}
    moved = canonical_regions(128, field_frac=0.5)
    with pytest.raises(SystemExit, match="detector layout differs"):
        check_layout(moved, 128, shipped)


def test_layout_guard_rejects_a_different_layout_at_a_different_grid():
    """The normalised comparison must not launder a real change through a rescale."""
    shipped = {"n": 128, "regions": canonical_regions(128)}
    moved = canonical_regions(256, field_frac=0.5)
    with pytest.raises(SystemExit, match="detector layout differs"):
        check_layout(moved, 256, shipped)


def test_layout_guard_rejects_a_different_class_count():
    shipped = {"n": 128, "regions": canonical_regions(128)}
    with pytest.raises(SystemExit, match="detector regions"):
        check_layout(canonical_regions(128)[:8], 128, shipped)


# ------------------------------------------------------------------------- tags

def test_config_tag_preserves_legacy_names_at_the_default_grid():
    """`z2mm_L14` is already in the results file, names a masks_*.npy, and is
    referenced by apps/export_sweep_web -- it must not gain a prefix."""
    assert config_tag(128, 2.0, 14) == "z2mm_L14"
    assert config_tag(128, 1.0345, 28) == "z1.0345mm_L28"


def test_config_tag_separates_other_grids():
    assert config_tag(256, 4.0, 14) == "n256_z4mm_L14"
    assert config_tag(256, 4.0, 14) != config_tag(128, 4.0, 14)


def test_dx_and_wavelength_are_the_shipped_operating_point():
    """The grid moves; the operating point does not."""
    assert DX == pytest.approx(8e-6)
    assert WAVELENGTH == pytest.approx(532e-9)
