"""Tests for input encoding and dataset splitting (photonn/train.py).

Only pure, in-memory operations are exercised here; the MNIST download in
``load_dataset`` is intentionally not part of the unit suite (no network in
tests), so ``split_dataset`` is checked against a synthetic Dataset.
"""
import numpy as np
import pytest
import torch

from photonn.train import Dataset, encode_input, split_dataset


def test_encode_schemes_shape_dtype_and_unit_norm():
    imgs = torch.rand(6, 28, 28)
    for scheme in ("amplitude", "phase", "both"):
        f = encode_input(imgs, scheme=scheme, n=64)
        assert f.shape == (6, 64, 64)
        assert f.dtype == torch.complex64
        l2 = f.abs().pow(2).sum(dim=(-2, -1))
        assert torch.allclose(l2, torch.ones(6), atol=1e-5)


def test_encode_amplitude_scheme_has_zero_phase():
    f = encode_input(torch.rand(3, 28, 28), scheme="amplitude", n=48)
    assert torch.allclose(f.imag, torch.zeros_like(f.imag), atol=1e-6)


def test_encode_phase_scheme_has_uniform_amplitude_in_window():
    f = encode_input(torch.rand(1, 28, 28), scheme="phase", n=48, input_frac=0.5)
    amp = f.abs()[0]
    illuminated = amp[amp > 0]
    # plane-wave through an aperture: every lit pixel shares one amplitude.
    assert torch.allclose(illuminated, illuminated.mean().expand_as(illuminated), atol=1e-5)


def test_encode_both_scheme_couples_amplitude_and_phase():
    # where the image is bright, both |E| and arg(E) should be non-trivial.
    img = torch.zeros(1, 28, 28)
    img[0, 10:18, 10:18] = 1.0
    f = encode_input(img, scheme="both", n=48, phase_scale=1.0)[0]
    lit = f.abs() > 0
    assert lit.any()
    assert f.angle()[lit].abs().max() > 0.0  # phase carries the image too


def test_encode_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        encode_input(torch.rand(1, 28, 28), scheme="hologram", n=32)


# -- train/val splitting -------------------------------------------------------
def _toy(n=200):
    rng = np.random.default_rng(0)
    return Dataset(images=rng.random((n, 28, 28), dtype=np.float32),
                   labels=rng.integers(0, 10, size=n), name="mnist", split="train")


def test_split_dataset_is_disjoint_and_covers_everything():
    """The whole point: no sample may appear in both halves.

    Two ``load_dataset(subset=...)`` calls would overlap heavily; this must not.
    """
    ds = _toy()
    train, val = split_dataset(ds, n_val=50, seed=7)

    assert len(train) == 150 and len(val) == 50
    keys = lambda d: {tuple(img.ravel()[:4]) for img in d.images}
    assert keys(train).isdisjoint(keys(val))
    assert len(keys(train) | keys(val)) == len(ds)


def test_split_dataset_is_deterministic_given_the_seed():
    ds = _toy()
    a, _ = split_dataset(ds, n_val=40, seed=3)
    b, _ = split_dataset(ds, n_val=40, seed=3)
    c, _ = split_dataset(ds, n_val=40, seed=4)

    assert np.array_equal(a.images, b.images)
    assert not np.array_equal(a.images, c.images)


def test_split_dataset_keeps_images_and_labels_aligned():
    """A permutation applied to one array and not the other would silently poison training."""
    ds = _toy(60)
    lookup = {tuple(img.ravel()[:4]): lab for img, lab in zip(ds.images, ds.labels)}
    for part in split_dataset(ds, n_val=20, seed=1):
        for img, lab in zip(part.images, part.labels):
            assert lookup[tuple(img.ravel()[:4])] == lab


@pytest.mark.parametrize("n_val", [0, 200, 500, -1])
def test_split_dataset_rejects_degenerate_sizes(n_val):
    with pytest.raises(ValueError, match="n_val"):
        split_dataset(_toy(), n_val=n_val)
