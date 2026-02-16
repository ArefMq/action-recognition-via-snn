import torch

from spikenet.constants import W_INIT_MEAN, W_INIT_STD
from spikenet.layers.neuron_base import NeuronBase


class _ConcreteNeuron(NeuronBase):
    """Minimal concrete subclass for testing."""

    def forward(self, x):
        return x


def test_default_initialization():
    neuron = _ConcreteNeuron()
    assert neuron.name == "NeuronBase"
    assert neuron.in_features is None
    assert neuron.out_features is None
    assert neuron.w_init_mean == W_INIT_MEAN
    assert neuron.w_init_std == W_INIT_STD


def test_custom_initialization():
    neuron = _ConcreteNeuron(
        name="Custom",
        in_features=10,
        out_features=20,
        w_init_mean=0.5,
        w_init_std=0.3,
    )
    assert neuron.name == "Custom"
    assert neuron.in_features == 10
    assert neuron.out_features == 20
    assert neuron.w_init_mean == 0.5
    assert neuron.w_init_std == 0.3


def test_params_empty_by_default():
    neuron = _ConcreteNeuron()
    assert neuron.params == []


def test_pytorch_parameters():
    """Registered parameters are accessible via PyTorch's .parameters()."""
    neuron = _ConcreteNeuron()
    neuron.weight = torch.nn.Parameter(torch.randn(3, 3))
    params = list(neuron.parameters())
    assert len(params) == 1
    assert isinstance(params[0], torch.nn.Parameter)


def test_forward_passes_through():
    neuron = _ConcreteNeuron()
    x = torch.randn(2, 3)
    output = neuron(x)
    assert torch.equal(output, x)
