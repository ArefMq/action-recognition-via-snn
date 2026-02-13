import torch

from spikenet.layers.flattening import Flatten


def test_forward_standard_shape():
    """(batch, channels, time, H, W) -> (batch, time, channels*H*W)."""
    layer = Flatten(out_features=4 * 7 * 7)
    x = torch.rand(2, 4, 10, 7, 7)
    output = layer(x)
    assert output.shape == (2, 10, 196)


def test_forward_single_channel():
    layer = Flatten(out_features=1 * 28 * 28)
    x = torch.rand(2, 1, 10, 28, 28)
    output = layer(x)
    assert output.shape == (2, 10, 784)


def test_forward_single_timestep():
    layer = Flatten(out_features=8 * 3 * 3)
    x = torch.rand(1, 8, 1, 3, 3)
    output = layer(x)
    assert output.shape == (1, 1, 72)


def test_forward_preserves_data():
    """All original values should be present after reshape."""
    layer = Flatten(out_features=2 * 2 * 2)
    x = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 2, 2)
    output = layer(x)
    assert set(output.flatten().tolist()) == set(x.flatten().tolist())
