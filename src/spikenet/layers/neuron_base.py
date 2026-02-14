from abc import ABC, abstractmethod

import torch
from loguru import logger

from spikenet.constants import W_INIT_MEAN, W_INIT_STD


class NeuronBase(torch.nn.Module, ABC):
    """
    Base class for all neuron layers used in the SpikeNet framework. This class is based on PyTorch's Module class,
    and provides a simpler interface with spiking specific methods.

    Args:
        name (str): Name of the layer
        in_features (int): Number of input features. None for getting the value from previous layer or input tensor.
        out_features (int): Number of output features.
        w_init_mean (float): Mean of the normal distribution used to initialize the weights.
        w_init_std (float): Standard deviation of the normal distribution used to initialize the weights.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.name = kwargs.get("name", "NeuronBase")
        self.in_features = kwargs.get("in_features")
        self.out_features = kwargs.get("out_features")
        self.w_init_mean = kwargs.get("w_init_mean", W_INIT_MEAN)
        self.w_init_std = kwargs.get("w_init_std", W_INIT_STD)

    @property
    def params(self) -> list[torch.nn.Parameter]:
        """Returns internal PyTorch parameters of this layer."""
        return [p for p in self.__dict__.values() if isinstance(p, torch.nn.Parameter)]

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the forward pass of the layer."""

    def initialize_parameters(self) -> None:
        """Initialise the parameters of the layer, usually with the internal mean and deviation"""

    def clamp(self) -> None:
        """Clamp the parameters of the layer to ensure they stay within valid ranges."""

    def reset(self) -> None:
        """Reset the layer to its initial state."""
        self.initialize_parameters()
        self.clamp()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ HELPERS ~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _assert(self, condition: bool, message: str, warning: bool = False) -> None:
        """Assert a condition and raise an error if it is not met."""
        if not warning:
            assert condition, f"[{self.name}] {message}"
        elif not condition:
            logger.warning(f"[{self.name}] {message}")

    def _check_nan(self, tensor: torch.Tensor, name: str, warning: bool = True) -> None:
        """Check if the tensor contains NaN values."""
        self._assert(not torch.isnan(tensor).any(), f"{name} contains NaN values", warning=warning)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ STRING REPRESENTATION ~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __repr__(self) -> str:
        return (
            f"{self.name}(in={self.in_features}, out={self.out_features}, "
            f"mean={self.w_init_mean}, std={self.w_init_std})"
        )

    def __str__(self) -> str:
        return f"{self.name}({self.out_features})"
