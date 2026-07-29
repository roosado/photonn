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
