from dataclasses import dataclass, field

import torch


@dataclass
class Criterion:
    optimizer_generator: torch.optim.Optimizer = torch.optim.SGD
    optimizer: torch.optim.Optimizer | None = None
    loss_fn: torch.nn.Module = torch.nn.NLLLoss
    encoding: torch.nn.Module = field(default=torch.nn.LogSoftmax(dim=1))
    epochs: int = 10
    learning_rate: float = 0.0001

    def get_optim(self, net: torch.nn.Module) -> torch.optim.Optimizer:
        if self.optimizer is not None:
            return self.optimizer
        return self.optimizer_generator(net.parameters(), lr=self.learning_rate, momentum=0.9)

    def get_loss_fn(self, _: torch.nn.Module) -> torch.nn.Module:
        return self.loss_fn()
