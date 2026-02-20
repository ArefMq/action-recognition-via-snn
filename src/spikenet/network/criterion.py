from dataclasses import dataclass, field

import torch

from spikenet.constants import DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_MOMENTUM, DEFAULT_WEIGHT_DECAY


@dataclass
class Criterion:
    optimizer_generator: torch.optim.Optimizer = torch.optim.SGD
    optimizer: torch.optim.Optimizer | None = None
    loss_fn: torch.nn.Module = torch.nn.NLLLoss
    encoding: torch.nn.Module = field(default=torch.nn.LogSoftmax(dim=1))
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    momentum: float = DEFAULT_MOMENTUM
    weight_decay: float = DEFAULT_WEIGHT_DECAY

    def get_optim(self, net: torch.nn.Module, lr: float | None = None) -> torch.optim.Optimizer:
        if self.optimizer is not None:
            return self.optimizer
        return self.optimizer_generator(
            net.parameters(),
            lr=lr or self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def get_loss_fn(self, _: torch.nn.Module) -> torch.nn.Module:
        return self.loss_fn()
