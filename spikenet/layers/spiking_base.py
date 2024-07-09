from abc import ABC, abstractmethod
from typing import Callable

import torch
from spikenet.functions import TimeReduction
from spikenet.layers.neuron_base import NeuronBase
from spikenet.tools.heaviside import SurrogateHeaviside


class SpikingNeuron(NeuronBase, ABC):
    """
    Base class for all spiking neuron layers used in the SpikeNet framework.

    Args:
        name (str): Name of the layer (default: <id>_<neuron_type>)
        in_features (int): Number of input features. None for getting the value from previous layer or input tensor.
        out_features (int): Number of output features.
        w_init_mean (float): Mean of the normal distribution used to initialize the weights.
        w_init_std (float): Standard deviation of the normal distribution used to initialize the weights.
        spike_fn (Callable): The spike function to use (default: SurrogateHeaviside.apply)
        time_reduction (str or TimeReduction): The time reduction method to use (default: TimeReduction.NoTimeReduction)
        beta_init_std (float): Standard deviation of the normal distribution used to initialize the beta parameter.
        beta_init_mean (float): Mean of the normal distribution used to initialize the beta parameter.
        b_init_std (float): Standard deviation of the normal distribution used to initialize the b parameter.
        b_init_mean (float): Mean of the normal distribution used to initialize the b parameter.

    NOTE: TimeReduction method is used to convert spiking activity to a single value. This is used either to
    feed the spiking network to a non-spiking network or to use the output of the spiking network in a decision-making
    process. The default value is TimeReduction.NoTimeReduction which means the output of the spiking network is the
    spikes themselves. Other options are:
        - SpikeRate: the output is the sum of spikes over time normalized by the number of time steps
        - SpikeTime: the output is the time of the first spike
        - MemRecMax: the output is the maximum value of the membrane potential over time
        - MemRecMean: the output is the mean value of the membrane potential over time

    NOTE: You can also provide a custom time reduction method by passing a callable function to the time_reduction argument.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.spike_fn = kwargs.get("spike_fn", SurrogateHeaviside.apply)
        self.__mem_rec: torch.Tensor | None = None
        self.__spike_rec: torch.Tensor | None = None
        time_reduction = kwargs.get("time_reduction", TimeReduction.NoTimeReduction)
        self.__time_reduction: TimeReduction | Callable = (
            TimeReduction(time_reduction)
            if isinstance(time_reduction, str)
            else time_reduction
        )
        self.w: torch.nn.Parameter | None = None
        self.beta: torch.nn.Parameter | None = None
        self.b: torch.nn.Parameter | None = None

        self.beta_init_std = kwargs.get("beta_init_std", 0.01)
        self.beta_init_mean = kwargs.get("beta_init_mean", 0.7)
        self.b_init_std = kwargs.get("b_init_std", 0.01)
        self.b_init_mean = kwargs.get("b_init_mean", 1.0)

    @property
    def is_spiking(self) -> bool:
        return True

    @property
    def time_reduction_method(self) -> str:
        return (
            self.__time_reduction.name
            if isinstance(self.__time_reduction, TimeReduction)
            else "CustomReduction"
        )

    @property
    def mem_rec(self) -> torch.Tensor:
        """
        Returns the membrane potential record of the neuron
        The membrane potential record is a tensor of shape (batch_size, time_steps, *out_features)
        """
        assert self.__mem_rec is not None, "mem_rec is not initialized"
        return self.__mem_rec

    @property
    def spike_rec(self) -> torch.Tensor:
        """
        Returns the spike record of the neuron
        The spike record is a binary tensor of shape (batch_size, time_steps, *out_features)
        """
        assert self.__spike_rec is not None, "spike_rec is not initialized"
        return self.__spike_rec

    @property
    def w_norm(self) -> torch.Tensor:
        assert self.w is not None, "Parameters w are not initialized"
        norm = (self.w**2).sum(0)
        assert not torch.isnan(norm).any()
        return norm

    def clamp(self) -> None:
        if self.beta is not None:
            self.beta.data.clamp_(0.0, 1.0)
        if self.b is not None:
            self.b.data.clamp_(min=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the neuron
        :param x: the input tensor
        :return: the output tensor
        """
        # FIXME: this assert is not working on pooling layer
        # assert self.w is not None, "Parameters are not initialized"
        # if not x.any():
        #     print(
        #         f"[Warning-{self.name}] No spikes in the layer when trying to forward"
        #     )
        spk_rec, mem_rec = self.spike_forward(x)
        # FIXME: this assert is not working on pooling layer
        # assert not torch.isnan(mem_rec).any(), f"[{self.name}] NaN in mem_rec"
        # assert not torch.isnan(spk_rec).any(), f"[{self.name}] NaN in spk_rec"
        self.__mem_rec = mem_rec
        res = self.time_reduction(spk_rec, mem_rec)
        self.__spike_rec = res
        assert not torch.isnan(res).any(), f"[{self.name}] NaN in output"
        return res

    @abstractmethod
    def spike_forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        The forward pass of the neuron
        :param x: the input tensor
        :return: the spikes and the membrane potential
        """
        ...

    @abstractmethod
    def initialize_parameters(self) -> None:
        """
        Initialize the parameters of the neuron
        """
        if self.w is not None or self.beta is not None or self.b is not None:
            print(f"[Warning-{self.name}] Parameters are already initialized")
        return super().initialize_parameters()

    # ~~~~~~~~ Time Reduction ~~~~~~~~
    def time_reduction(self, spk: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        if (
            self.__time_reduction is None
            or self.__time_reduction == TimeReduction.NoTimeReduction
        ):
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
        return torch.nn.functional.one_hot(
            max_min_data, num_classes=self.out_features
        ).to(torch.float32)

    def __time_reduction_mem_rec_max(self, mem_rec: torch.Tensor) -> torch.Tensor:
        output = torch.max(mem_rec, 1)[0] / (self.w_norm + 1e-8) - self.b
        return output

    def __time_reduction_mem_rec_mean(self, mem_rec: torch.Tensor) -> torch.Tensor:
        output = torch.mean(mem_rec, 1) / (self.w_norm + 1e-8) - self.b
        return output

    def details(self) -> str:
        txt = super().details()
        if (
            self.__time_reduction is None
            or self.__time_reduction == TimeReduction.NoTimeReduction
        ):
            time_reduction = ""
        else:
            time_reduction = f" TimeReduction: {self.time_reduction_method}"
        return f"spk({txt}){time_reduction}"
