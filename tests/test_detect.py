"""Tests for detector regions, readout, and the photon budget (photonn/detect.py)."""
import numpy as np
import pytest

from photonn.fields import Field
from photonn import detect

LAM = 532e-9


def test_default_regions_count_within_grid_and_nonoverlapping():
    n = 128
    regions = detect.default_regions(n, 10)
    assert len(regions) == 10

    canvas = np.zeros((n, n), dtype=int)
    for r in regions:
        assert 0 <= r.y0 < r.y1 <= n
        assert 0 <= r.x0 < r.x1 <= n
        canvas[r.y0:r.y1, r.x0:r.x1] += 1
    assert canvas.max() == 1  # no pixel is claimed by two regions


def test_integrate_intensity_matches_manual_sum():
    n, dx = 32, 8e-6
    rng = np.random.default_rng(3)
    f = Field(rng.random((n, n)), dx, LAM)
    regions = detect.default_regions(n, 4)

    got = detect.integrate_intensity(f, regions)
    manual = np.array([f.intensity()[r.slices].sum() * dx**2 for r in regions])
    assert np.allclose(got, manual)


def test_photon_budget_energy_accounting():
    n, dx = 16, 8e-6
    f = Field(np.ones((n, n)), dx, LAM)
    pb = detect.photon_budget(f, input_power_w=1e-3, integration_time_s=1e-6,
                              reference_power=f.power())

    e_ph = detect.H_PLANCK * detect.C_LIGHT / LAM
    assert pb.photon_energy_j == pytest.approx(e_ph)
    assert pb.photons_in == pytest.approx(1e-3 * 1e-6 / e_ph)
    # reference_power == input power => the per-pixel photon map sums to N_in.
    assert pb.per_pixel.sum() == pytest.approx(pb.photons_in, rel=1e-9)


def test_photon_budget_captured_fraction_in_unit_interval():
    n, dx = 64, 8e-6
    rng = np.random.default_rng(4)
    f = Field(rng.random((n, n)), dx, LAM)
    regions = detect.default_regions(n, 10)

    pb = detect.photon_budget(f, input_power_w=1e-3, integration_time_s=1e-6,
                              regions=regions, reference_power=f.power())
    assert pb.per_region.shape == (10,)
    assert 0.0 <= pb.captured_fraction <= 1.0


def test_detector_region_rejects_empty_and_negative():
    with pytest.raises(ValueError):
        detect.DetectorRegion(5, 5, 0, 3)   # empty in y
    with pytest.raises(ValueError):
        detect.DetectorRegion(0, 3, -1, 3)  # negative bound


# -- the readout contract, across runtimes -------------------------------------

def test_numpy_and_torch_readouts_agree():
    """`detect.region_logits` is the reference the torch readout is a port of.

    The formula lives in four runtimes -- NumPy here, torch in `models.D2NN`,
    MATLAB in `+model/readout.m`, JavaScript in `d2nn.js` -- and they must agree,
    because `readout_gain` crosses the handoff as a number whose meaning *is* this
    formula. Nothing pinned any two of them to each other; each was internally
    self-consistent and that was all.
    """
    import torch

    from photonn.detect import default_regions, region_logits
    from photonn.fields import Field
    from photonn.models import D2NN

    n, gain = 32, 10.0
    rng = np.random.default_rng(20260904)
    data = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))

    regions = default_regions(n, 10)
    model = D2NN(n_layers=1, n=n, dx=8e-6, wavelength=532e-9, separation=1e-3,
                 regions=regions, readout_gain=gain)

    # The torch readout, fed the same field, with propagation taken out of it.
    # readout_masks is float32; match it so einsum has one dtype.
    x = torch.as_tensor(data, dtype=torch.complex64)[None, ...]
    intensity = x.real ** 2 + x.imag ** 2
    region = torch.einsum("bij,cij->bc", intensity, model.readout_masks)
    total = intensity.sum(dim=(-2, -1)).clamp_min(1e-12).unsqueeze(-1)
    torch_logits = (region / total * model.readout_gain)[0].numpy()

    numpy_logits = region_logits(Field(data, 8e-6, 532e-9), regions, gain)
    # rel=1e-6: the torch side carries float32 masks, so this is that dtype's
    # precision rather than any looseness about the formula.
    assert numpy_logits == pytest.approx(torch_logits, rel=1e-6, abs=1e-12)


def test_the_readout_is_invariant_to_uniform_loss():
    """Why the formula normalises by total power at all.

    Insertion loss scales the whole field. If the readout did not divide by total
    power, every logit would fall together and an as-built run would look less
    confident rather than unchanged -- and the error budget would be reading a
    scale factor as damage.
    """
    from photonn.detect import default_regions, region_logits
    from photonn.fields import Field

    n = 32
    rng = np.random.default_rng(7)
    data = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    regions = default_regions(n, 10)

    full = region_logits(Field(data, 8e-6, 532e-9), regions, 10.0)
    lossy = region_logits(Field(0.1 * data, 8e-6, 532e-9), regions, 10.0)
    assert full == pytest.approx(lossy, rel=1e-12)
