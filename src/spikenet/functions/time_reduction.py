from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from spikenet.constants import EPSILON

if TYPE_CHECKING:
    from spikenet.layers.spiking_base import SpikingNeuron

TimeReductionFunction = Callable[["SpikingNeuron", Tensor, Tensor], Tensor]


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
    """Reduce the spike recording based on the time of the spikes. Each neuron's score is a
    weighted sum of its spikes, where earlier spikes receive a higher weight (T - t). A neuron
    that fires at t=0 scores T; one that never fires scores 0. This preserves the first-to-spike
    semantics while keeping the output differentiable through spk_rec.

    Args:
        neuron: Spiking neuron
        spk_rec: Spike recordings with shape (batch_size, time_steps, num_neurons)

    Returns:
        Early-spike scores with shape (batch_size, num_neurons)
    """
    T = spk_rec.shape[1]
    time_weights = T - torch.arange(T, dtype=spk_rec.dtype, device=spk_rec.device)
    return torch.einsum("btn,t->bn", spk_rec, time_weights)


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
    return torch.max(mem_rec, 1)[0] / (neuron.w_norm + EPSILON) - neuron.b


def mean_membrane_potential(neuron: SpikingNeuron, _: Tensor, mem_rec: Tensor) -> Tensor:
    """Reduce the membrane potential recording based on the mean value of the membrane potential.

    Args:
        neuron: Spiking neuron
        mem_rec: Membrane potential recordings with shape (batch_size, time_steps, num_neurons)

    Returns:
        Mean membrane potentials with shape (batch_size, num_neurons)
    """
    return torch.mean(mem_rec, 1) / (neuron.w_norm + EPSILON) - neuron.b
