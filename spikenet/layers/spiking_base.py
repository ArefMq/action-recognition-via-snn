from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable

import numpy as np
import torch
from spikenet.layers.neuron_base import NeuronBase
from spikenet.tools.heaviside import SurrogateHeaviside


class TimeReduction(Enum):
    NoTimeReduction = None
    SpikeRate = "SpikeRate"
    SpikeTime = "SpikeTime"
    MemRecMax = "MemRecMax"
    MemRecMean = "MemRecMean"


class SpikingNeuron(NeuronBase, ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.spike_fn = kwargs.get("spike_fn", SurrogateHeaviside.apply)
        self.__mem_rec: torch.Tensor = None
        self.__spike_rec: torch.Tensor = None
        time_reduction = kwargs.get("time_reduction", TimeReduction.NoTimeReduction)
        self.__time_reduction: TimeReduction | Callable = (
            TimeReduction(time_reduction)
            if isinstance(time_reduction, str)
            else time_reduction
        )
        self.w: torch.nn.Parameter = None
        self.beta: torch.nn.Parameter = None
        self.b: torch.nn.Parameter = None

    @property
    def is_spiking(self) -> bool:
        return True

    @property
    def mem_rec(self) -> torch.Tensor:
        """
        Returns the membrane potential record of the neuron
        The membrane potential record is a tensor of shape (batch_size, time_steps, *output_dim)
        """
        return self.__mem_rec

    @property
    def spike_rec(self) -> torch.Tensor:
        """
        Returns the spike record of the neuron
        The spike record is a binary tensor of shape (batch_size, time_steps, *output_dim)
        """
        return self.__spike_rec

    @property
    def w_norm(self) -> torch.Tensor:
        norm = (self.w**2).sum(0)
        assert not torch.isnan(norm).any()
        return norm

    def clamp(self) -> None:
        self.beta.data.clamp_(0.0, 1.0)
        self.b.data.clamp_(min=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the neuron
        :param x: the input tensor
        :return: the output tensor
        """
        assert self.w is not None, "Parameters are not initialized"
        if not x.any():
            print(
                f"[Warning-{self.name}] No spikes in the layer when trying to forward"
            )
        spk_rec, mem_rec = self.spike_forward(x)
        assert not torch.isnan(mem_rec).any(), f"[{self.name}] NaN in mem_rec"
        assert not torch.isnan(spk_rec).any(), f"[{self.name}] NaN in spk_rec"
        self.__mem_rec = mem_rec
        self.__spike_rec = spk_rec
        res = self.time_reduction(spk_rec, mem_rec)
        assert not torch.isnan(res).any(), f"[{self.name}] NaN in output"
        return res

    @abstractmethod
    def spike_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass of the neuron
        :param x: the input tensor
        :return: the spikes and the membrane potential
        """
        ...

    @abstractmethod
    def initialize_parameters(self):
        """
        Initialize the parameters of the neuron
        """
        if self.w is not None or self.beta is not None or self.b is not None:
            print(f"[Warning-{self.name}] Parameters are already initialized")
        return super().initialize_parameters()

    # ~~~~~~~~ Time Reduction ~~~~~~~~
    def time_reduction(self, spk: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        if self.__time_reduction is None or self.__time_reduction == TimeReduction.NoTimeReduction:
            return spk
        if callable(self.__time_reduction):
            return self.__time_reduction(spk)
        match self.__time_reduction:
            case TimeReduction.SpikeRate:
                return self.__time_reduction_spike_rate(spk)
            case TimeReduction.SpikeTime:
                return self.__time_reduction_spike_time(spk)
            case TimeReduction.MemRecMax:
                return self.__time_reduction_mem_rec_max(mem)
            case TimeReduction.MemRecMean:
                return self.__time_reduction_mem_rec_mean(mem)
        raise ValueError(f"Invalid time_reduction {self.__time_reduction}")

    def __time_reduction_spike_rate(self, x: torch.Tensor) -> torch.Tensor:
        spike_rate = torch.sum(x, 1)
        return torch.nn.functional.softmax(spike_rate, dim=1)

    def __time_reduction_spike_time(self, x: torch.Tensor) -> torch.Tensor:
        max_data = torch.max(x, 1).indices
        max_min_data = max_data.min(1)[1]
        return torch.nn.functional.one_hot(max_min_data, num_classes=self.output_dim).to(torch.float32)

    def __time_reduction_mem_rec_max(self, mem_rec: torch.Tensor) -> torch.Tensor:
        output = torch.max(mem_rec, 1)[0] / (self.w_norm + 1e-8) - self.b
        return output
    
    def __time_reduction_mem_rec_mean(self, mem_rec: torch.Tensor) -> torch.Tensor:
        output = torch.mean(mem_rec, 1) / (self.w_norm + 1e-8) - self.b
        return output

