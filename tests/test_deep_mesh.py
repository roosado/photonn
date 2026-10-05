"""Phase 5: the activated deep mesh, its NumPy reference, and its handoff (schema 0.4.0).

The torch model (models.DeepMeshNetwork) is held to a float64 NumPy forward pass built
from the same arrays a handoff stores (mzi.deep_mesh_forward), which is what the MATLAB
as-built model and the browser are held to in turn. Power is physical from here on, so
two tests are about units rather than numbers: the activation must see watts, and a
pre-activation Sigma must stay passive.
"""
from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from apps.train_deep_mesh import deep_parameters
from photonn import mzi
from photonn.export import OPERATING_POINT, validate_handoff, write_handoff
from photonn.handoff import read_handoff, read_parameters, read_test_inputs
from photonn.layers import MZIMeshLayer
from photonn.models import DeepMeshNetwork
from photonn.train import encode_fourier, fourier_order

_REPO = Path(__file__).resolve().parent.parent
_DEEP_H5 = _REPO / "exports" / "deep_mesh_phase5.h5"
_DEEP_PT = _REPO / "exports" / "deep_mesh_phase5.pt"

P_IN, G, ALPHA, PHI_B = 1e-3, 0.05 * math.pi / 1e-3, 0.1, math.pi


def _model(n=6, seed=3, **kw):
    torch.manual_seed(seed)
    m = DeepMeshNetwork(n, 4, n_layers=2, alpha=ALPHA, g_phi=G, phi_b=PHI_B,
                        input_power_w=kw.pop("input_power_w", P_IN), **kw)
    # Move sigma off its initial value so the test is not of a trivial layer.
    with torch.no_grad():
        for layer in m.layers:
            layer.sigma.add_(torch.randn_like(layer.sigma) * 0.5)
    return m


def _unit_inputs(n, b=32, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((b, n)) + 1j * rng.standard_normal((b, n))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# -- the NumPy reference ------------------------------------------------------------
def test_schedule_and_brick_match_the_torch_mesh():
    layer = MZIMeshLayer(7)
    assert mzi.clements_schedule(7) == layer._schedule
    with torch.no_grad():
        layer.out_phase.uniform_(0, 2 * math.pi)
    m = mzi.brick_matrix(layer.theta.detach().double().numpy(),
                         layer.phi.detach().double().numpy(),
                         layer.out_phase.detach().double().numpy())
    assert np.allclose(m, layer.matrix().detach().numpy(), atol=2e-6)


def test_torch_model_matches_the_float64_reference():
    m = _model()
    params, gain = deep_parameters(m)
    x = _unit_inputs(6)
    logits_t = m(torch.as_tensor(x, dtype=torch.complex64)).detach().numpy()
    out, _ = mzi.deep_mesh_forward(params, x, input_power_w=P_IN, alpha=ALPHA, g_phi=G,
                                   phi_b=PHI_B)
    intensity = np.abs(out) ** 2
    logits_n = intensity[:, :4] / intensity.sum(axis=1, keepdims=True) * m.readout_gain
    assert np.allclose(logits_t, logits_n, atol=5e-5)
    assert gain > 0                          # the external gain passivize hands back


def test_the_activation_sees_watts_and_they_scale_with_input_power():
    """Plan 12's trap: a unit-norm field fed to Eq. (6) silently sits the device at 1 W."""
    x = torch.as_tensor(_unit_inputs(6), dtype=torch.complex64)
    p1 = _model(input_power_w=1e-3).activation_input_power_w(x)[0]
    p10 = _model(input_power_w=1e-2).activation_input_power_w(x)[0]
    assert torch.allclose(p10, 10 * p1, rtol=1e-5)
    # ...and the bank sees at most the input power: everything ahead of it is passive.
    assert float(p1.sum(dim=1).max()) <= 1e-3 * (1 + 1e-5)
    params, _ = deep_parameters(_model(input_power_w=1e-3))
    _, taps = mzi.deep_mesh_forward(params, x.numpy().astype(complex), input_power_w=1e-3,
                                    alpha=ALPHA, g_phi=G, phi_b=PHI_B)
    assert np.allclose(taps[0], p1.numpy(), rtol=1e-4, atol=1e-12)


def test_carrying_unit_norm_is_the_same_as_carrying_watts():
    """The model writes g*P on the unit-norm field; that must equal sqrt(P) scaling."""
    x = _unit_inputs(5)
    z_w = mzi.eo_activation(math.sqrt(P_IN) * x, alpha=ALPHA, g_phi=G, phi_b=PHI_B)
    z_u = mzi.eo_activation(x, alpha=ALPHA, g_phi=G * P_IN, phi_b=PHI_B)
    assert np.allclose(z_w, math.sqrt(P_IN) * z_u, rtol=1e-12, atol=0)


def test_sigma_ahead_of_the_activation_is_passive_and_is_exported_unfolded():
    """passivize() upstream of an activation would be wrong; the exporter must not do it."""
    m = _model()
    with torch.no_grad():
        m.layers[0].sigma.fill_(50.0)          # sigmoid saturates at 1, never above
    params, _ = deep_parameters(m)
    assert params["sigma"][0].max() <= 1.0
    assert np.allclose(params["sigma"][0], torch.sigmoid(m.layers[0].sigma).detach().double().numpy())
    assert not m.layers[-1].passive and m.layers[0].passive


# -- the input ------------------------------------------------------------------------
def test_fourier_encoder_is_the_papers_definition():
    rng = np.random.default_rng(1)
    img = rng.random((28, 28)).astype(np.float32)
    v = encode_fourier(img, n_modes=16)[0].numpy().astype(complex)
    m = np.arange(28)
    k = np.fft.fftfreq(28) * 2 * np.pi
    c = np.array([[np.sum(np.exp(1j * kx * m[:, None] + 1j * ky * m[None, :]) * img)
                   for ky in k] for kx in k])
    ref = c.ravel()[fourier_order(28)[:16]]
    assert np.allclose(v, ref / np.linalg.norm(ref), atol=1e-6)
    assert abs(np.linalg.norm(v) - 1) < 1e-6


def test_fourier_order_takes_the_smallest_k_first():
    k = np.fft.fftfreq(28) * 28
    order = fourier_order(28)[:16]
    k2 = [k[i // 28] ** 2 + k[i % 28] ** 2 for i in order]
    assert k2 == sorted(k2) and k2[0] == 0 and max(k2) == 5


# -- the handoff ------------------------------------------------------------------------
def test_deep_mesh_roundtrip(tmp_path, deep_mesh_payload):
    path = tmp_path / "deep.h5"
    write_handoff(path, **deep_mesh_payload)
    validate_handoff(path)
    h = read_handoff(path)
    assert h.model_type == "deep_mesh" and h.schema_version == "0.4.0"
    p = read_parameters(path)
    for key in ("phase_theta", "phase_phi", "sigma", "out_phase"):
        assert np.allclose(p[key], deep_mesh_payload["parameters"][key])
    assert int(p["n_layers"]) == 2
    assert np.allclose(read_test_inputs(path), deep_mesh_payload["test_inputs"])
    assert h.op("eo_g_phi") == pytest.approx(deep_mesh_payload["operating_point"]["eo_g_phi"])


@pytest.mark.parametrize("dropped", sorted(
    k for k, spec in OPERATING_POINT.items() if "deep_mesh" in spec.required_for))
def test_every_required_deep_mesh_constant_is_enforced(tmp_path, deep_mesh_payload, dropped):
    deep_mesh_payload["operating_point"].pop(dropped)
    with pytest.raises(ValueError, match=dropped):
        write_handoff(tmp_path / "x.h5", **deep_mesh_payload)


def test_derived_constants_must_follow_from_the_device(tmp_path, deep_mesh_payload):
    deep_mesh_payload["operating_point"]["eo_g_phi"] *= 1.01
    with pytest.raises(ValueError, match="eo_g_phi"):
        write_handoff(tmp_path / "x.h5", **deep_mesh_payload)


def test_a_sigma_above_one_is_refused(tmp_path, deep_mesh_payload):
    deep_mesh_payload["parameters"]["sigma"][0, 0] = 1.2
    with pytest.raises(ValueError, match="passive"):
        write_handoff(tmp_path / "x.h5", **deep_mesh_payload)


def test_a_deep_mesh_without_its_inputs_is_refused(tmp_path, deep_mesh_payload):
    deep_mesh_payload.pop("test_inputs")
    with pytest.raises(ValueError, match="test_inputs"):
        write_handoff(tmp_path / "x.h5", **deep_mesh_payload)


def test_older_mesh_files_have_no_inputs_and_say_so(tmp_path, mesh_payload):
    path = tmp_path / "mesh.h5"
    write_handoff(path, **mesh_payload)
    assert read_test_inputs(path) is None


# -- the trained model, when it is on disk -----------------------------------------------
@pytest.mark.skipif(not _DEEP_H5.exists(), reason="exports/deep_mesh_phase5.h5 not present")
def test_exported_model_reproduces_its_stated_accuracy_in_float64():
    h = read_handoff(_DEEP_H5)
    p = read_parameters(_DEEP_H5)
    x = read_test_inputs(_DEEP_H5)
    with h5py.File(_DEEP_H5, "r") as f:
        labels = f["test_set/labels"][...]
    out, _ = mzi.deep_mesh_forward(p, x, input_power_w=h.op("input_power_w"),
                                   alpha=h.op("eo_alpha"), g_phi=h.op("eo_g_phi"),
                                   phi_b=h.op("eo_phi_b"))
    acc = float(np.mean((np.abs(out[:, :10]) ** 2).argmax(axis=1) == labels))
    assert acc == h.test_acc


@pytest.mark.skipif(not (_DEEP_H5.exists() and _DEEP_PT.exists()),
                    reason="Phase-5 exports not present")
def test_handoff_rebuilds_the_checkpoints_logits():
    ckpt = torch.load(_DEEP_PT, weights_only=False)
    h = read_handoff(_DEEP_H5)
    m = DeepMeshNetwork(16, 10, n_layers=2, alpha=h.op("eo_alpha"), g_phi=h.op("eo_g_phi"),
                        phi_b=h.op("eo_phi_b"), input_power_w=h.op("input_power_w"))
    m.load_state_dict(ckpt["state_dict"])
    x = read_test_inputs(_DEEP_H5)[:500]
    logits_t = m(torch.as_tensor(x, dtype=torch.complex64)).detach().numpy()
    out, _ = mzi.deep_mesh_forward(read_parameters(_DEEP_H5), x,
                                   input_power_w=h.op("input_power_w"),
                                   alpha=h.op("eo_alpha"), g_phi=h.op("eo_g_phi"),
                                   phi_b=h.op("eo_phi_b"))
    intensity = np.abs(out) ** 2
    logits_n = intensity[:, :10] / intensity.sum(axis=1, keepdims=True) * 10.0
    # float32 torch against float64 NumPy, logits up to ~10: measured 1.2e-4. The cubic
    # triples the field's relative round-off, so this is looser than the mesh's.
    assert np.abs(logits_t - logits_n).max() < 5e-4
