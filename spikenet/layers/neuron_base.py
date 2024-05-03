from abc import ABC, abstractmethod

import torch

from spikenet.tools.configs import W_INIT_MEAN, W_INIT_STD


class NeuronBase(torch.nn.Module, ABC):
    @classmethod
    def __layer_id(cls) -> int:
        if not hasattr(cls, "_id"):
            cls._id = 0
        cls._id += 1
        return cls._id

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.name = kwargs.get("name", f"{self.__layer_id()}_{self.__class__.__name__}")
        self.in_features = kwargs["in_features"]
        if self.in_features is not None:
            self.in_features = int(self.in_features)
        self.out_features = int(kwargs["out_features"])
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

    def __str__(self) -> str:
        return f"{self.name}({self.out_features})"

