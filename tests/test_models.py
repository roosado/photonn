"""Tests for the D2NN model (photonn/models.py)."""
import torch

from photonn.models import D2NN
from photonn.train import encode_input


def _model(n, n_layers):
    return D2NN(n=n, n_layers=n_layers, dx=8e-6, wavelength=532e-9, separation=3e-3)


def test_d2nn_forward_shape_and_gradients_flow():
    n = 32
    model = _model(n, n_layers=3)
    field = encode_input(torch.rand(5, 28, 28), scheme="both", n=n)

    logits = model(field)
    assert logits.shape == (5, 10)

    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3, 4]))
    loss.backward()
    for m in model.masks:
        assert m.phi.grad is not None
        assert torch.isfinite(m.phi.grad).all()


def test_d2nn_only_phase_masks_are_trainable():
    n = 16
    model = _model(n, n_layers=4)
    # propagation transfer functions and readout masks are buffers, not parameters.
    assert sum(p.numel() for p in model.parameters()) == 4 * n * n


def test_d2nn_output_field_is_complex_and_shape_preserving():
    n = 24
    model = _model(n, n_layers=2)
    field = encode_input(torch.rand(2, 28, 28), scheme="both", n=n)
    out = model.output_field(field)
    assert out.shape == (2, n, n)
    assert out.is_complex()


def test_d2nn_logits_are_scale_invariant():
    # normalising by total output power => logits unchanged if the input is scaled.
    n = 24
    model = _model(n, n_layers=2)
    field = encode_input(torch.rand(3, 28, 28), scheme="both", n=n)
    a = model(field)
    b = model(field * 7.0)
    assert torch.allclose(a, b, atol=1e-4)
