from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
import torch
from torch.nn.parameter import Parameter

from spikenet.dataloader import DataLoader


def _default_call_back(net, epoch, i, loss):
    if i == 0:
        print(f"\nEpoch: {epoch}: ", end="")
    if i % 20 != 0:
        return
    print(".", end="")


@dataclass
class Network:
    @dataclass
    class Criterion:
        optimizer_generator: torch.optim.Optimizer = torch.optim.SGD
        optimizer: torch.optim.Optimizer | Any | None = None
        loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss
        epochs: int = 10
        learning_rate: float = 0.01

        def get_optim(self, net: torch.nn.Module) -> torch.optim.Optimizer:
            if self.optimizer is not None:
                return self.optimizer
            return self.optimizer_generator(net.parameters(), lr=self.learning_rate)

        def get_loss_fn(self, net: torch.nn.Module) -> torch.nn.Module:
            return self.loss_fn()

        def obsorb_parameters(self, kwargs: dict[str, Any]) -> dict[str, Any]:
            res = {}
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    res[key] = value
            return res

    name: str = "My Network"
    layers: torch.nn.ModuleList = field(default_factory=torch.nn.ModuleList)
    device: torch.device = field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    criterion: Criterion = field(default_factory=Criterion)

    def add_layer(
        self, module: torch.nn.Module, *args, **kwargs
    ) -> "Network":
        self.layers.append(module(*args, **kwargs))
        return self

    @classmethod
    def from_parameters(cls, *args, **kwargs) -> "Network":
        cri = cls.Criterion()
        kwargs = cri.obsorb_parameters(kwargs)
        net = cls(*args, criterion=cri, **kwargs)
        return net

    @classmethod
    def from_layers(cls, layers: torch.nn.ModuleList, *args, **kwargs) -> "Network":
        cri = cls.Criterion()
        kwargs = cri.obsorb_parameters(kwargs)
        net = cls(*args, criterion=cri, **kwargs)
        net.layers = layers
        return net

    def build(self) -> "Network._CompiledNetwork":
        return Network._CompiledNetwork(self).to(self.device)

    def fit(self, dataloader: DataLoader, **kwargs) -> "Network._CompiledNetwork":
        # fix the input size of the first layer and then build the network
        net = self.build()
        net.initialize_parameters()
        a, b = net.fit(dataloader, **kwargs)
        return net

    class _CompiledNetwork(torch.nn.Module):
        def __init__(self, config: "Network"):
            super().__init__()
            self.config = config
            self.initialize_parameters()

        @property
        def layers(self) -> torch.nn.ModuleList:
            return self.config.layers

        def initialize_parameters(self) -> None:
            for layer in self.layers:
                layer.initialize_parameters()

        def parameters(self) -> Iterator[Parameter]:
            for layer in self.layers:
                for param in layer.parameters():
                    yield param

        def forward(self, x: torch.Tensor, save_history: bool = True) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x, save_history=save_history)
            return x

        def fit(self, dataloader: DataLoader, **kwargs) -> tuple[int, int]:
            self.train(dataloader, **kwargs)
            return self.test(dataloader)

        def train(
            self, dataloader: DataLoader | bool, callback: Callable | None = "default", **kwargs
        ) -> None:
            if callback == "default":
                callback = _default_call_back
            if isinstance(dataloader, bool):
                return super().train(dataloader)

            kwargs = self.config.criterion.obsorb_parameters(kwargs)
            crit = self.config.criterion

            # get criterion functions
            optimizer = crit.get_optim(self)
            loss_fn = crit.get_loss_fn(self)

            for epoch in range(crit.epochs):
                for i, (x_data, y_data) in enumerate(dataloader("train")):
                    x_data: torch.Tensor = x_data.to(self.config.device)
                    y_data: torch.Tensor = y_data.to(self.config.device)

                    assert not torch.isnan(x_data).any(), "NaN in x_data"
                    assert not torch.isnan(y_data).any(), "NaN in y_data"

                    # Forward pass
                    outputs = self.forward(x_data, save_history=False)
                    assert not torch.isnan(outputs).any(), "NaN in outputs"
                    loss: torch.Tensor = loss_fn(outputs, y_data)
                    assert not torch.isnan(loss).any(), "NaN in loss"

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if callback is not None:
                        callback(self, epoch, i, loss.item())
                self.snapshot_cycle()
        
        def snapshot_cycle(self) -> None:
            for layer in self.layers:
                layer.snapshot_cycle()

        def test(
            self, dataloader: DataLoader, callback: Callable | None = "default"
        ) -> tuple[int, int]:
            if callback == "default":
                callback = _default_call_back
            self.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for i, (x_data, y_data) in enumerate(dataloader("test")):
                    x_data = x_data.to(self.config.device)
                    y_data = y_data.to(self.config.device)

                    outputs = self(x_data)
                    # TODO: this should be a function
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_data.size(0)
                    correct += (predicted == y_data).sum().item()

                    if callback is not None:
                        callback(self, i, total, correct)
            return total, correct

        def __repr__(self):
            return f"{self.config.name}:\n{self.layers}"
