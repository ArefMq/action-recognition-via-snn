from abc import ABC, abstractmethod

import torch

from spikenet.tools.configs import W_INIT_MEAN, W_INIT_STD


class NeuronBase(torch.nn.Module, ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.name = kwargs.get('name', None)
        self.input_dim = int(kwargs["input_dim"])
        self.output_dim = int(kwargs["output_dim"])
        self.w_init_mean = kwargs.get('w_init_mean', W_INIT_MEAN)
        self.w_init_std = kwargs.get('w_init_std', W_INIT_STD)

    @property
    def params(self) -> list[torch.nn.Parameter]:
        return [p for p in self.__dict__.values() if isinstance(p, torch.nn.Parameter)]

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def initialize_parameters(self):
        pass

    def clamp(self):
        pass

    def reset(self):
        self.initialize_parameters()
        self.clamp()
