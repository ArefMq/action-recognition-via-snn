import torch

from spikenet.layers.spiking_dense import SpikingDenseLayer
from spikenet.tools.time_reduction import spike_rate


def _make_layer(in_features=10, out_features=5, **kwargs):
    layer = SpikingDenseLayer(in_features=in_features, out_features=out_features, **kwargs)
    layer.initialize_parameters()
    return layer


def test_initialize_parameters_shapes():
    layer = _make_layer()
    assert layer.w.shape == (10, 5)
    assert layer.beta.shape == (1,)
    assert layer.b.shape == (5,)
    assert layer.w.requires_grad
    assert layer.beta.requires_grad
    assert layer.b.requires_grad


def test_spike_forward_output_shapes():
    layer = _make_layer()
    x = torch.rand(2, 8, 10)
    spk_rec, mem_rec = layer.spike_forward(x)
    assert spk_rec.shape == (2, 8, 5)
    assert mem_rec.shape == (2, 8, 5)


def test_spike_forward_binary_spikes():
    layer = _make_layer()
    x = (torch.rand(2, 8, 10) > 0.5).float()
    spk_rec, _ = layer.spike_forward(x)
    assert ((spk_rec == 0) | (spk_rec == 1)).all()


def test_spike_forward_zero_input():
    layer = _make_layer()
    x = torch.zeros(2, 8, 10)
    spk_rec, mem_rec = layer.spike_forward(x)
    assert spk_rec.sum() == 0
    assert mem_rec.sum() == 0


def test_spike_forward_membrane_evolves():
    layer = _make_layer()
    x = (torch.rand(2, 20, 10) > 0.3).float()
    _, mem_rec = layer.spike_forward(x)
    # With non-zero input over enough timesteps, membrane should be non-trivial
    assert mem_rec.abs().sum() > 0


def test_forward_no_time_reduction():
    layer = _make_layer()
    x = torch.rand(2, 8, 10)
    output = layer(x)
    assert output.shape == (2, 8, 5)


def test_forward_spike_rate_reduction():
    layer = _make_layer(time_reduction=spike_rate)
    x = (torch.rand(2, 8, 10) > 0.5).float()
    output = layer(x)
    assert output.shape == (2, 5)
    # spike_rate applies softmax, so each sample sums to ~1
    assert torch.allclose(output.sum(dim=1), torch.ones(2), atol=1e-5)


def test_gradient_flow():
    layer = _make_layer()
    x = torch.rand(2, 8, 10)
    output = layer(x)
    loss = output.sum()
    loss.backward()
    assert layer.w.grad is not None
    assert layer.beta.grad is not None
    assert layer.b.grad is not None


def test_single_timestep():
    layer = _make_layer()
    x = torch.rand(1, 1, 10)
    spk_rec, mem_rec = layer.spike_forward(x)
    assert spk_rec.shape == (1, 1, 5)
    assert mem_rec.shape == (1, 1, 5)
