from abc import ABC, abstractmethod

import torch
from torch import Tensor

from spikenet.layers.neuron_base import NeuronBase
from spikenet.tools.heaviside import SurrogateHeaviside
from spikenet.tools.time_reduction import TimeReductionFunction, no_time_reduction


class SpikingNeuron(NeuronBase, ABC):
    """
    Base class for all spiking neuron layers used in the SpikeNet framework.

    Args:
        name (str): Name of the layer (default: "SpikingNeuron")
        in_features (int): Number of input features. None for getting the value from previous layer or input tensor.
        out_features (int): Number of output features.
        w_init_mean (float): Mean of the normal distribution used to initialize the weights.
        w_init_std (float): Standard deviation of the normal distribution used to initialize the weights.
        spike_fn (Callable): The spike function to use (default: SurrogateHeaviside.apply)
        time_reduction (Callable | None): The time reduction method to use (default: None)
        beta_init_std (float): Standard deviation of the normal distribution used to initialize the beta parameter.
        beta_init_mean (float): Mean of the normal distribution used to initialize the beta parameter.
        b_init_std (float): Standard deviation of the normal distribution used to initialize the b parameter.
        b_init_mean (float): Mean of the normal distribution used to initialize the b parameter.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(name=kwargs.pop("name", "SpikingNeuron"), **kwargs)
        self.spike_fn = kwargs.get("spike_fn", SurrogateHeaviside.apply)
        self.time_reduction_fn: TimeReductionFunction = kwargs.get("time_reduction", no_time_reduction)

        # Internal attributes .....................
        self.__mem_rec: Tensor | None = None
        self.__spike_rec: Tensor | None = None

        # Learning weights and parameters ........
        self.__w: torch.nn.Parameter | None = None
        self.__beta: torch.nn.Parameter | None = None
        self.__b: torch.nn.Parameter | None = None

        # Initialization parameters ..............
        self.beta_init_std = kwargs.get("beta_init_std", 0.01)
        self.beta_init_mean = kwargs.get("beta_init_mean", 0.7)
        self.b_init_std = kwargs.get("b_init_std", 0.01)
        self.b_init_mean = kwargs.get("b_init_mean", 1.0)

    @property
    def w(self) -> torch.nn.Parameter:
        """Weight parameter of the neuron."""
        assert self.__w is not None, "Weight parameter is not initialized"
        return self.__w

    @w.setter
    def w(self, value: torch.nn.Parameter) -> None:
        """Set the weight parameter of the neuron."""
        assert self.__w is None, "Weight parameter is already initialized"
        self.__w = value

    @property
    def beta(self) -> torch.nn.Parameter:
        """Beta parameter of the neuron."""
        assert self.__beta is not None, "Beta parameter is not initialized"
        return self.__beta

    @beta.setter
    def beta(self, value: torch.nn.Parameter) -> None:
        """Set the beta parameter of the neuron."""
        assert self.__beta is None, "Beta parameter is already initialized"
        self.__beta = value

    @property
    def b(self) -> torch.nn.Parameter:
        """B parameter of the neuron."""
        assert self.__b is not None, "B parameter is not initialized"
        return self.__b

    @b.setter
    def b(self, value: torch.nn.Parameter) -> None:
        """Set the b parameter of the neuron."""
        assert self.__b is None, "B parameter is already initialized"
        self.__b = value

    @property
    def mem_rec(self) -> Tensor:
        """Membrane potential record of the neuron.

        Returns:
            a tensor of shape (batch_size, time_steps, *out_features)

        Raises:
            AssertionError: if mem_rec is not initialized
        """
        assert self.__mem_rec is not None, "mem_rec is not initialized"
        return self.__mem_rec

    @property
    def spike_rec(self) -> Tensor:
        """Spike record of the neuron.

        Returns:
            a binary tensor of shape (batch_size, time_steps, *out_features)

        Raises:
            AssertionError: if spike_rec is not initialized
        """
        assert self.__spike_rec is not None, "spike_rec is not initialized"
        return self.__spike_rec

    @property
    def w_norm(self) -> Tensor:
        """Weight norm of the neuron.

        Returns:
            a tensor of shape (out_features,)

        Raises:
            AssertionError: if w is not initialized
            AssertionError: if w_norm contains NaN values
        """
        assert self.w is not None, "Parameters w are not initialized"
        norm = (self.w**2).sum(0)
        self._check_nan(norm, "w_norm")
        return norm

    def clamp(self) -> None:
        """Implementation of the parameter clamping for the neuron."""
        if self.beta is not None:
            self.beta.data.clamp_(0.0, 1.0)
        if self.b is not None:
            self.b.data.clamp_(min=0.0)

    def forward(self, x: Tensor) -> Tensor:
        """The forward pass of the neuron.

        Args:
            x (Tensor): the input tensor value

        Returns:
            Tensor: the output tensor
        """
        assert self.w is not None, "Parameters are not initialized"
        self._assert(not x.any(), "No input spikes.")
        spk_rec, mem_rec = self.spike_forward(x)
        self._check_nan(mem_rec, "mem_rec")
        self._check_nan(spk_rec, "spk_rec")

        self.__mem_rec = mem_rec
        reduced_spike_rec = self.time_reduction_fn(self, spk_rec, mem_rec)
        self.__spike_rec = reduced_spike_rec
        self._check_nan(reduced_spike_rec, "output")
        return reduced_spike_rec

    @abstractmethod
    def spiking_forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        """Spiking specific implementation of the forward pass.

        Args:
            x (Tensor): the input tensor

        Returns:
            tuple[Tensor, Tensor | None]: the spikes and the membrane potential
        """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HELPERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def avg_mem(self) -> float:
        """Return the average membrane potential. Used to debug layer activity."""
        return self.__mem_rec.mean().item()

    def avg_spike(self) -> float:
        """Return the average spike count. Used to debug layer activity."""
        return self.__spike_rec.mean().item()

    def mem_percentage(self) -> float:
        """Return the percentage of membrane potential. Used to debug layer activity."""
        return (self.__mem_rec > 0).sum().item() / self.__mem_rec.numel()

    def spike_percentage(self) -> float:
        """Return the percentage of spike count. Used to debug layer activity."""
        return (self.__spike_rec > 0).sum().item() / self.__spike_rec.numel()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ STRING REPRESENTATION ~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __repr__(self) -> str:
        return (
            f"{self.name}(in={self.in_features}, out={self.out_features}, "
            f"mean={self.w_init_mean}, std={self.w_init_std}, "
            f"spike_fn={self.spike_fn.__name__}"
            f", time_reduction={self.time_reduction_method})"
            if self.time_reduction_method
            else ")"
        )

    def __str__(self) -> str:
        time_reduction = f", reduction={self.time_reduction_method.name}" if self.time_reduction_method else ""
        return f"{self.name}({self.out_features}, spiking{time_reduction})"
