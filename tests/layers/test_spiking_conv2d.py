import numpy as np
import torch

from spikenet.layers.spiking_conv2d import SpikingConv2D


def _make_layer(in_features=1, out_features=4, **kwargs):
    layer = SpikingConv2D(in_features=in_features, out_features=out_features, **kwargs)
    layer.initialize_parameters()
    return layer


def test_default_conv_params():
    layer = _make_layer()
    np.testing.assert_array_equal(layer.stride, [1, 1, 1])
    np.testing.assert_array_equal(layer.padding, [0, 0, 0])
    np.testing.assert_array_equal(layer.dilation, [1, 1, 1])
    np.testing.assert_array_equal(layer.kernel, [1, 3, 3])


def test_scalar_kernel():
    layer = _make_layer(kernel=5)
    np.testing.assert_array_equal(layer.kernel, [1, 5, 5])


def test_initialize_parameters_shapes():
    layer = _make_layer(kernel=(1, 3, 3))
    assert layer.w.shape == (4, 1, 1, 3, 3)
    assert layer.beta.shape == (1,)
    assert layer.b.shape == (4,)
    assert layer.w.requires_grad


def test_spike_forward_output_shapes():
    layer = _make_layer()
    x = torch.rand(2, 1, 8, 14, 14)
    spk_rec, mem_rec = layer.spike_forward(x)
    # default kernel (1,3,3) with no padding shrinks spatial dims by 2
    assert spk_rec.shape == (2, 4, 8, 12, 12)
    assert mem_rec.shape == (2, 4, 8, 12, 12)


def test_spike_forward_binary_spikes():
    layer = _make_layer()
    x = (torch.rand(2, 1, 8, 12, 12) > 0.5).float()
    spk_rec, _ = layer.spike_forward(x)
    assert ((spk_rec == 0) | (spk_rec == 1)).all()


def test_apply_conv_preserves_spatial():
    """With default kernel (1,3,3) and auto padding, spatial dims are preserved."""
    layer = _make_layer()
    x = torch.rand(2, 1, 8, 14, 14)
    conv_out = layer._apply_conv(x)
    assert conv_out.shape[0] == 2  # batch
    assert conv_out.shape[1] == 4  # out_features
    assert conv_out.shape[2] == 8  # time


def test_non_square_input():
    layer = _make_layer()
    x = torch.rand(1, 1, 4, 10, 16)
    spk_rec, mem_rec = layer.spike_forward(x)
    # kernel (1,3,3): spatial dims shrink by 2 each
    assert spk_rec.shape == (1, 4, 4, 8, 14)
    assert mem_rec.shape == (1, 4, 4, 8, 14)


def test_gradient_flow():
    layer = _make_layer()
    x = torch.rand(1, 1, 4, 10, 10)
    spk_rec, _ = layer.spike_forward(x)
    loss = spk_rec.sum()
    loss.backward()
    assert layer.w.grad is not None
