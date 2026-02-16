import torch

from spikenet.layers.spiking_dense import SpikingDenseLayer
from spikenet.network import Network


def test_add_layer_returns_self():
    net = Network()
    layer = SpikingDenseLayer(in_features=10, out_features=5)
    result = net.add_layer(layer)
    assert result is net


def test_add_layer_chaining():
    net = Network()
    l1 = SpikingDenseLayer(in_features=10, out_features=5)
    l2 = SpikingDenseLayer(in_features=5, out_features=3)
    net.add_layer(l1).add_layer(l2)
    assert len(net._layers) == 2


def test_forward_passes_through_layers():
    net = Network()
    l1 = SpikingDenseLayer(in_features=10, out_features=5)
    l1.initialize_parameters()
    net.add_layer(l1)

    x = torch.rand(2, 8, 10)
    output = net(x)
    assert output.shape == (2, 8, 5)


def test_initialize_parameters():
    net = Network()
    layer = SpikingDenseLayer(in_features=10, out_features=5)
    net.add_layer(layer)

    assert layer.w is None
    net.initialize_parameters()
    assert layer.w is not None
    assert layer.w.shape == (10, 5)


def test_parameters_yields_all():
    net = Network()
    layer = SpikingDenseLayer(in_features=10, out_features=5)
    layer.initialize_parameters()
    net.add_layer(layer)

    params = list(net.parameters())
    # w, beta, b
    assert len(params) == 3


def test_clamp_propagates():
    net = Network()
    layer = SpikingDenseLayer(in_features=10, out_features=5)
    layer.initialize_parameters()
    layer.beta.data.fill_(2.0)
    net.add_layer(layer)

    net.clamp()
    assert layer.beta.item() <= 1.0
