"""Invariants the readout re-score rests on (issue #5).

``apps/score_readout.py`` scores 48 detector layouts against **one** cached
forward pass instead of running the model 48 times, and it ranks them on data
that is neither trained on nor the frozen test set. Three things have to hold for
that shortcut and that protocol to mean anything:

* **The out-of-model readout must agree with the model's own.** Scoring happens
  from :meth:`~photonn.models.D2NN.output_field` and a stacked matmul, precisely
  so the ``readout_masks`` buffer -- which ``load_state_dict`` overwrites -- is
  never touched. If that path disagreed with ``D2NN.forward``, every number in
  the study would be measuring something other than the classifier.
* **The selection set must be disjoint from the frozen 2 000.** Choosing a layout
  is model selection, and ``photonn.train.split_dataset``'s docstring is explicit
  that model selection must not read the images every published accuracy is
  quoted from.
* **The layout grid must contain the shipped layout**, or "the shipped one wins"
  is a comparison against a set it does not belong to.

Pure Python and cheap -- no trained model, no gitignored file. The model here is
untrained (random masks); every property under test is a property of the scoring
path, not of what the masks learned.
"""
import numpy as np
import pytest
import torch

from apps.score_readout import (FROZEN_N, FROZEN_SEED, PATCH_FRACS, SHIPPED, holdout_split,
                                layout_grid, overlap_stats, score_layouts)
from photonn.detect import default_regions
from photonn.models import D2NN
from photonn.train import Dataset, encode_input, load_dataset

GRID = 32          # small enough to build a model per test; the geometry is fractional


def tiny_model(n=GRID, n_layers=2):
    torch.manual_seed(0)
    return D2NN(n=n, n_layers=n_layers, dx=8e-6, wavelength=532e-9,
                separation=3e-3, mask_init_std=0.5).eval()


def synthetic(n_images, seed):
    rng = np.random.default_rng(seed)
    return Dataset(images=rng.random((n_images, 28, 28)).astype("float32"),
                   labels=rng.integers(0, 10, n_images), name="mnist", split="synthetic")


# ------------------------------------------------------- the scoring path itself

def test_cached_scoring_reproduces_the_models_own_forward_pass():
    """The whole study is one forward pass plus a matmul; it must match ``forward``.

    ``D2NN.forward`` divides by total power and multiplies by ``readout_gain``;
    both are monotone per sample, so they cannot move an argmax -- which is why
    the scorer is allowed to skip them. This asserts that equivalence on
    predictions rather than assuming it.
    """
    model = tiny_model()
    ds = synthetic(16, seed=0)
    layouts = [({"shipped": True}, model.regions)]

    acc, _ = score_layouts(model, ds, layouts, scheme="both", batch=8)
    with torch.no_grad():
        field = encode_input(ds.images, scheme="both", n=model.n)
        direct = model(field).argmax(dim=1).numpy()

    assert acc[0] == (direct == ds.labels).mean()


def test_score_layouts_matches_a_direct_per_layout_evaluation():
    """The stacked 48-way matmul must equal scoring each layout on its own."""
    model = tiny_model()
    ds = synthetic(24, seed=1)

    layouts = layout_grid(model.n, field_fracs=(0.65, 0.75), patch_fracs=(0.11, 0.20))
    acc, cap = score_layouts(model, ds, layouts, scheme="both", batch=8)

    with torch.no_grad():
        out = model.output_field(encode_input(ds.images, scheme="both", n=model.n))
        i2 = (out.real ** 2 + out.imag ** 2).numpy()
    total = i2.reshape(len(ds), -1).sum(axis=1)

    for k, (_, regions) in enumerate(layouts):
        sums = np.stack([[i2[b][r.slices].sum() for r in regions] for b in range(len(ds))])
        assert acc[k] == (sums.argmax(axis=1) == ds.labels).mean()
        # Accuracy is a count and must match exactly; the captured fraction is a
        # float32 sum reduced in a different order, so it matches to float32 eps.
        assert cap[k] == pytest.approx(np.mean(sums.sum(axis=1) / total), rel=1e-6)


# -------------------------------------------------------------- the protocol

def test_the_selection_set_never_touches_the_frozen_test_set():
    """Ranking layouts on the frozen 2 000 would tune the design on the test set."""
    holdout = holdout_split()
    frozen = load_dataset("mnist", subset=FROZEN_N, split="test", subset_seed=FROZEN_SEED)

    assert len(holdout) + len(frozen) == 10_000
    # Compare on content, not indices: the holdout is built as an index complement,
    # so an index-level check would only restate its own construction.
    holdout_rows = {h.tobytes() for h in holdout.images}
    assert not any(f.tobytes() in holdout_rows for f in frozen.images)


def test_the_grid_contains_the_shipped_layout_exactly_once():
    shipped = [row for row, _ in layout_grid(128) if row["shipped"]]
    assert len(shipped) == 1
    assert (shipped[0]["field_frac"], shipped[0]["patch_frac"]) == SHIPPED
    assert shipped[0]["patch_px"] == 14          # 0.11 * 128, the shipped 14 px box


def test_the_grid_spans_both_sides_of_the_overlap_threshold():
    """A grid entirely inside the non-overlapping regime could not find the cliff."""
    rows = [row for row, _ in layout_grid(128)]
    assert len(rows) == 48
    assert any(r["max_cover"] == 1 for r in rows)
    assert any(r["max_cover"] > 1 for r in rows)


# ------------------------------------------------------------------ diagnostics

def test_overlap_stats_report_no_double_counting_at_the_shipped_layout():
    stats = overlap_stats(default_regions(128, 10), 128)
    assert stats["max_cover"] == 1
    assert stats["double_counted_frac"] == 0.0


def test_capture_exceeds_one_exactly_when_boxes_overlap():
    """Captured fraction > 1 *is* double counting, not a bug in the accounting.

    The same photon is summed by more than one class, so the ten region sums add
    to more than the power on the plane. Measuring that directly is what replaced
    the original study's inference from the accuracy fall-off alone. Overlap is
    necessary but not sufficient -- a barely-overlapping layout can still capture
    under 1 -- so the check is one-sided plus the extreme case.
    """
    model = tiny_model()
    ds = synthetic(8, seed=2)

    layouts = layout_grid(model.n, field_fracs=(0.75,), patch_fracs=PATCH_FRACS)
    _, cap = score_layouts(model, ds, layouts, scheme="both", batch=8)

    for (row, _), c in zip(layouts, cap):
        if row["max_cover"] == 1:
            assert c <= 1.0 + 1e-9, f"{row} captured {c} without overlapping"
    assert layouts[-1][0]["max_cover"] > 1 and cap[-1] > 1.0     # the widest box, 0.40
