import numpy as np
import torch

from spikenet.functions.pooling_reduction import avg_spike_rate
from spikenet.layers.spiking_pooling import SpikingPoolingLayer


def test_default_pooling_params():
    layer = SpikingPoolingLayer()
    np.testing.assert_array_equal(layer.stride, [1, 2, 2])
    np.testing.assert_array_equal(layer.kernel, [1, 2, 2])


def test_max_pooling_output_shape():
    layer = SpikingPoolingLayer()
    x = torch.rand(2, 4, 8, 14, 14)
    spk_rec, mem_rec = layer.spike_forward(x)
    assert spk_rec.shape == (2, 4, 8, 7, 7)
    assert mem_rec is None


def test_max_pooling_preserves_spikes():
    """Max pooling of a known binary pattern."""
    layer = SpikingPoolingLayer(kernel=(1, 2, 2), stride=(1, 2, 2))
    x = torch.zeros(1, 1, 1, 4, 4)
    x[0, 0, 0, 0, 0] = 1.0  # top-left 2x2 window
    x[0, 0, 0, 3, 3] = 1.0  # bottom-right 2x2 window
    spk_rec, _ = layer.spike_forward(x)
    assert spk_rec.shape == (1, 1, 1, 2, 2)
    assert spk_rec[0, 0, 0, 0, 0] == 1.0
    assert spk_rec[0, 0, 0, 1, 1] == 1.0
    assert spk_rec[0, 0, 0, 0, 1] == 0.0
    assert spk_rec[0, 0, 0, 1, 0] == 0.0


def test_avg_pooling():
    layer = SpikingPoolingLayer(reduction=avg_spike_rate)
    x = torch.ones(1, 1, 1, 4, 4)
    spk_rec, _ = layer.spike_forward(x)
    assert spk_rec.shape == (1, 1, 1, 2, 2)
    assert torch.allclose(spk_rec, torch.ones(1, 1, 1, 2, 2))


def test_avg_pooling_half_ones():
    """Half the values in each window are 1, so avg should be 0.5."""
    layer = SpikingPoolingLayer(reduction=avg_spike_rate, kernel=(1, 2, 2), stride=(1, 2, 2))
    x = torch.zeros(1, 1, 1, 2, 2)
    x[0, 0, 0, 0, 0] = 1.0
    x[0, 0, 0, 1, 1] = 1.0
    spk_rec, _ = layer.spike_forward(x)
    assert spk_rec.shape == (1, 1, 1, 1, 1)
    assert torch.isclose(spk_rec[0, 0, 0, 0, 0], torch.tensor(0.5))
