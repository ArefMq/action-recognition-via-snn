import pytest
import torch

from spikenet.layers.spiking_dense import SpikingDenseLayer
from spikenet.tools.heaviside import SurrogateHeaviside
from spikenet.tools.time_reduction import no_time_reduction


def _make_layer(in_features=10, out_features=5, **kwargs):
    layer = SpikingDenseLayer(in_features=in_features, out_features=out_features, **kwargs)
    layer.initialize_parameters()
    return layer


def test_default_spike_fn():
    layer = _make_layer()
    assert layer.spike_fn == SurrogateHeaviside.apply


def test_default_time_reduction():
    layer = _make_layer()
    assert layer.time_reduction_fn is no_time_reduction


def test_default_init_params():
    layer = _make_layer()
    assert layer.beta_init_mean == 0.7
    assert layer.beta_init_std == 0.01
    assert layer.b_init_mean == 1.0
    assert layer.b_init_std == 0.01


def test_w_is_none_before_init():
    layer = SpikingDenseLayer(in_features=10, out_features=5)
    assert layer.w is None


def test_mem_rec_before_forward_raises():
    layer = _make_layer()
    with pytest.raises(AssertionError, match="mem_rec is not initialized"):
        _ = layer.mem_rec


def test_spike_rec_before_forward_raises():
    layer = _make_layer()
    with pytest.raises(AssertionError, match="spike_rec is not initialized"):
        _ = layer.spike_rec


def test_clamp_beta_and_b():
    layer = _make_layer()
    layer.beta.data.fill_(1.5)
    layer.b.data.fill_(-0.5)
    layer.clamp()
    assert layer.beta.item() <= 1.0
    assert (layer.b >= 0.0).all()


def test_w_norm_shape_and_nonneg():
    layer = _make_layer()
    norm = layer.w_norm
    assert norm.shape == (5,)
    assert (norm >= 0).all()


def test_forward_populates_mem_and_spike_rec():
    layer = _make_layer()
    x = torch.rand(2, 8, 10)
    layer(x)
    assert layer.mem_rec.shape == (2, 8, 5)
    assert layer.spike_rec.shape == (2, 8, 5)
