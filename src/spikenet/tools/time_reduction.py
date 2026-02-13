from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from spikenet.layers.spiking_base import SpikingNeuron


def no_time_reduction(neuron: SpikingNeuron, spk_rec: Tensor, mem_rec: Tensor) -> Tensor:
    """No time reduction. Returns the input as is."""
    return spk_rec


def spike_rate(neuron: SpikingNeuron, spk_rec: Tensor, _: Tensor) -> Tensor:
    """Reduce spike recording based on the rate of the spikes. In other words, the neuron that fired more frequently
    through out the time, would fire. This methods uses softmax to select the firing neuron.

    Args:
        neuron: Spiking neuron
        spk_rec: Spike recordings with shape (batch_size, time_steps, num_neurons)

    Returns:
        Spike rates with shape (batch_size, num_neurons)
    """
    spike_rate = torch.sum(spk_rec, 1)
    return torch.nn.functional.softmax(spike_rate, dim=1)


def spike_time(neuron: SpikingNeuron, spk_rec: Tensor, _: Tensor) -> Tensor:
    """Reduce the spike recording based on the time of the spikes. In other words, the neuron that fired the earliest
    through out the time, would fire. This methods uses the index of the maximum value to select the firing neuron.

    Args:
    neuron: Spiking neuron
        spk_rec: Spike recordings with shape (batch_size, time_steps, num_neurons)

    Returns:
        Spike times with shape (batch_size, num_neurons)
    """
    max_data = torch.max(spk_rec, 1).indices
    max_min_data = max_data.min(1)[1]
    return torch.nn.functional.one_hot(max_min_data, num_classes=neuron.out_features).to(torch.float32)


def max_membrane_potential(neuron: SpikingNeuron, _: Tensor, mem_rec: Tensor) -> Tensor:
    """Reduce the membrane potential recording based on the maximum value of the membrane potential.
    NOTE: Compared to `spike_rate` and `spike_time`, this method is not relying on the spiking activity of the neurons,
    hence it can be argued that the network is a spiking based.

    Args:
        neuron: Spiking neuron
        mem_rec: Membrane potential recordings with shape (batch_size, time_steps, num_neurons)

    Returns:
        Maximum membrane potentials with shape (batch_size, num_neurons)
    """
    return torch.max(mem_rec, 1)[0] / (neuron.w_norm + 1e-8) - neuron.b


def mean_membrane_potential(neuron: SpikingNeuron, _: Tensor, mem_rec: Tensor) -> Tensor:
    """Reduce the membrane potential recording based on the mean value of the membrane potential.

    Args:
        neuron: Spiking neuron
        mem_rec: Membrane potential recordings with shape (batch_size, time_steps, num_neurons)

    Returns:
        Mean membrane potentials with shape (batch_size, num_neurons)
    """
    return torch.mean(mem_rec, 1) / (neuron.w_norm + 1e-8) - neuron.b
