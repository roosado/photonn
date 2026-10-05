"""Analytic tests for the Phase-5 electro-optic activation (photonn/mzi.py).

Williamson et al., IEEE JSTQE 26(1):7700412 (2020), Eq. (5)-(7). The correctness
anchor is the identity with an MZI built from this project's own primitives
(validate.eo_activation_reference) -- the activation's counterpart of Clements
reconstruction. The rest are the closed-form limits plan 12 names.
"""
import math

import numpy as np
import pytest
import torch

from photonn import mzi, validate
from photonn.layers import EOActivationLayer


def _draws(n, seed=20261004):
    rng = np.random.default_rng(seed)
    z = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * rng.uniform(0, 3, n)
    alpha = rng.uniform(0.0, 0.5, n)
    g = rng.uniform(0.0, 2.0, n)
    phi_b = rng.uniform(0.0, 2 * np.pi, n)
    return z, alpha, g, phi_b


def test_closed_form_is_our_own_mzi():
    """Eq. (6) equals sqrt(1-alpha)[B P(theta) B]_10 z at theta = -(phi_b + g|z|^2)."""
    z, alpha, g, phi_b = _draws(2000)
    worst = 0.0
    for zi, a, gi, pb in zip(z, alpha, g, phi_b):
        f = mzi.eo_activation(zi, alpha=a, g_phi=gi, phi_b=pb)
        ref = validate.eo_activation_reference(zi, alpha=a, g_phi=gi, phi_b=pb)
        worst = max(worst, abs(f - ref) / max(abs(zi), 1e-300))
    # Measured 2.6e-14: these draws write up to ~200 rad on the modulator, and that is
    # float64 round-off in cos() of a large argument, not a disagreement.
    assert worst < 1e-13


def test_small_signal_limit_at_pi_bias_is_a_pure_cubic():
    """phi_b = pi, g|z|^2 << 1: f -> -sqrt(1-alpha)(g/2)|z|^2 z, no linear term."""
    alpha, g = 0.1, 0.05 * math.pi
    z = np.array([1e-3, 2e-3 + 1e-3j, -3e-3j]) * 1.0
    f = mzi.eo_activation(z, alpha=alpha, g_phi=g, phi_b=math.pi)
    cubic = -math.sqrt(1 - alpha) * (g / 2) * np.abs(z) ** 2 * z
    assert np.allclose(f, cubic, rtol=1e-6, atol=0)


def test_zero_gain_is_a_constant_linear_element():
    """g = 0: the device is a fixed attenuator sqrt(1-alpha) cos(phi_b/2), times a phase."""
    z, alpha, _, phi_b = _draws(200)
    for zi, a, pb in zip(z, alpha, phi_b):
        f = mzi.eo_activation(zi, alpha=a, g_phi=0.0, phi_b=pb)
        expect = 1j * math.sqrt(1 - a) * math.cos(pb / 2) * np.exp(-0.5j * pb) * zi
        assert f == pytest.approx(expect, rel=1e-13, abs=1e-15)


def test_fully_open_at_pi_bias_when_the_light_writes_pi():
    alpha, g = 0.1, 0.3
    p = math.pi / g
    f = mzi.eo_activation(math.sqrt(p), alpha=alpha, g_phi=g, phi_b=math.pi)
    assert abs(f) ** 2 / p == pytest.approx(1 - alpha, rel=1e-12)


def test_activation_is_passive_everywhere_sampled():
    z, alpha, g, phi_b = _draws(5000, seed=7)
    for zi, a, gi, pb in zip(z, alpha, g, phi_b):
        f = mzi.eo_activation(zi, alpha=a, g_phi=gi, phi_b=pb)
        assert abs(f) ** 2 <= (1 - a) * abs(zi) ** 2 * (1 + 1e-12)


def test_passivity_assertion_fires_on_an_amplifying_output():
    with pytest.raises(ValueError, match="not passive"):
        mzi.assert_passive(np.array([1.0]), np.array([1.0]), alpha=0.1)
    with validate.relaxed():
        mzi.assert_passive(np.array([1.0]), np.array([1.0]), alpha=0.1)


def test_alpha_outside_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="alpha"):
        mzi.eo_activation(1.0, alpha=1.0, g_phi=0.1, phi_b=0.0)


def test_phase_gain_and_bias_are_equations_7_and_5():
    g = mzi.eo_phase_gain(alpha=0.1, tia_gain_ohm=5000.0, responsivity_a_per_w=1.0, v_pi=10.0)
    assert g == pytest.approx(math.pi * 0.1 * 5000.0 * 1.0 / 10.0)
    assert mzi.eo_bias_phase(v_bias=10.0, v_pi=10.0) == pytest.approx(math.pi)


# -- the torch layer ---------------------------------------------------------------
def test_torch_layer_matches_numpy_reference():
    z, *_ = _draws(500, seed=11)
    layer = EOActivationLayer(alpha=0.1, g_phi=0.7, phi_b=2.1)
    out = layer(torch.as_tensor(z, dtype=torch.complex128)).numpy()
    ref = mzi.eo_activation(z, alpha=0.1, g_phi=0.7, phi_b=2.1)
    assert np.allclose(out, ref, rtol=0, atol=1e-13)


@pytest.mark.parametrize("phi_b", [math.pi, 0.4])
def test_torch_layer_gradient_matches_finite_differences(phi_b):
    """The trap the gate fell into: torch.polar's backward is wrong for a signed modulus.

    At phi_b = pi the modulus cos((g|z|^2 + pi)/2) is negative throughout the weak-light
    tail, and polar() takes the sign of its result as the direction of d/d|r|. Forward
    agrees to 1e-13 either way; only a gradient check tells them apart.
    """
    gen = torch.Generator().manual_seed(0)
    re = (torch.rand(40, dtype=torch.float64, generator=gen) * 2).requires_grad_()
    im = (torch.rand(40, dtype=torch.float64, generator=gen) * 2).requires_grad_()
    layer = EOActivationLayer(alpha=0.1, g_phi=0.05 * math.pi, phi_b=phi_b)

    def f(a, b):
        out = layer(torch.complex(a, b))
        return out.real, out.imag

    assert torch.autograd.gradcheck(f, (re, im))


def test_torch_polar_really_does_get_the_signed_modulus_wrong():
    """Canary: if torch fixes polar's backward, the comment in layers.py can go."""
    a = torch.tensor([-0.3], dtype=torch.float64, requires_grad=True)
    th = torch.tensor([0.7], dtype=torch.float64)
    torch.polar(a, th).real.sum().backward()
    assert a.grad.item() == pytest.approx(-math.cos(0.7))     # analytic answer is +cos(0.7)
