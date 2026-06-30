import torch

from spikenet.functions.time_reduction import (
    max_membrane_potential,
    mean_membrane_potential,
    no_time_reduction,
    spike_rate,
    spike_time,
)
from spikenet.layers.spiking_dense import SpikingDenseLayer


def _make_layer(in_features=10, out_features=5):
    layer = SpikingDenseLayer(in_features=in_features, out_features=out_features)
    layer.initialize_parameters()
    return layer


def test_no_time_reduction_returns_spk_rec():
    layer = _make_layer()
    spk_rec = torch.rand(2, 8, 5)
    mem_rec = torch.rand(2, 8, 5)
    result = no_time_reduction(layer, spk_rec, mem_rec)
    assert torch.equal(result, spk_rec)


def test_spike_rate_output_shape():
    layer = _make_layer()
    spk_rec = (torch.rand(2, 8, 5) > 0.5).float()
    mem_rec = torch.rand(2, 8, 5)
    result = spike_rate(layer, spk_rec, mem_rec)
    assert result.shape == (2, 5)


def test_spike_rate_sums_to_one():
    """spike_rate applies softmax, so each sample should sum to ~1."""
    layer = _make_layer()
    spk_rec = (torch.rand(4, 8, 5) > 0.5).float()
    result = spike_rate(layer, spk_rec, torch.zeros(4, 8, 5))
    assert torch.allclose(result.sum(dim=1), torch.ones(4), atol=1e-5)


def test_spike_time_output_is_one_hot():
    layer = _make_layer()
    spk_rec = (torch.rand(2, 8, 5) > 0.7).float()
    result = spike_time(layer, spk_rec, torch.zeros(2, 8, 5))
    assert result.shape == (2, 5)
    # each row should have exactly one 1.0
    assert torch.equal(result.sum(dim=1), torch.ones(2))


def test_max_membrane_potential_output_shape():
    layer = _make_layer()
    x = torch.rand(2, 8, 10)
    _, mem_rec = layer.spike_forward(x)
    result = max_membrane_potential(layer, torch.zeros(2, 8, 5), mem_rec)
    assert result.shape == (2, 5)


def test_mean_membrane_potential_output_shape():
    layer = _make_layer()
    x = torch.rand(2, 8, 10)
    _, mem_rec = layer.spike_forward(x)
    result = mean_membrane_potential(layer, torch.zeros(2, 8, 5), mem_rec)
    assert result.shape == (2, 5)
