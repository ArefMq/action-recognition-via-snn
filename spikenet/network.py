from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
import torch
from torch.nn.parameter import Parameter

from spikenet.dataloader import DataLoader
from spikenet.tools.utils.callbacks_helper import CallbackFactory, CallbackTypes


@dataclass
class Network:
    @dataclass
    class Criterion:
        optimizer_generator: torch.optim.Optimizer = torch.optim.SGD
        optimizer: torch.optim.Optimizer | Any | None = None
        loss_fn: torch.nn.Module = torch.nn.L1Loss
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
        self,
        module: torch.nn.Module,
        output_dim: int,
        input_dim: int | None = None,
        **kwargs,
    ) -> "Network":
        if input_dim is None and len(self.layers) > 0:
            input_dim = self.layers[-1].output_dim
        self.layers.append(module(input_dim=input_dim, output_dim=output_dim, **kwargs))
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
        for layer in self.layers:
            layer.to(self.device)
            assert layer.input_dim is not None, f"input_dim is not set for {layer}"
        return Network._CompiledNetwork(self).to(self.device)

    def fit(self, dataloader: DataLoader, **kwargs) -> "Network._CompiledNetwork":
        if self.layers[0].input_dim is None:
            input_shape, _ = dataloader.shape
            # FIXME : this will not work with conv.layers
            self.layers[0].input_dim = input_shape[1]
        net = self.build()
        net.fit(dataloader, **kwargs)
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x)
            return x

        def clamp(self) -> None:
            for layer in self.layers:
                layer.clamp()

        def fit(self, dataloader: DataLoader, **kwargs) -> "Network._CompiledNetwork":
            self.train(dataloader, **kwargs)
            self.test(dataloader)
            return self

        def train(
            self,
            dataloader: DataLoader | bool,
            callbacks: CallbackTypes = "default",
            **kwargs,
        ) -> None:
            callbacks = CallbackFactory.parse_callbacks(callbacks)
            if isinstance(dataloader, bool):
                return super().train(dataloader)

            kwargs = self.config.criterion.obsorb_parameters(kwargs)
            crit = self.config.criterion

            # get criterion functions
            optimizer = crit.get_optim(self)
            loss_fn = crit.get_loss_fn(self)

            callbacks(net=self, call_type="train.before")
            for epoch in range(crit.epochs):
                callbacks(
                    net=self,
                    epoch=epoch,
                    dataload_length=dataloader.len("train"),
                    call_type="train.epoch.before",
                )
                for i, (x_data, y_data) in enumerate(dataloader("train")):
                    callbacks(
                        net=self,
                        epoch=epoch,
                        batch_id=i,
                        call_type="train.batch.before",
                    )

                    x_data: torch.Tensor = x_data.to(self.config.device)
                    y_data: torch.Tensor = y_data.to(self.config.device)

                    assert not torch.isnan(x_data).any(), "NaN in x_data"
                    assert not torch.isnan(y_data).any(), "NaN in y_data"

                    # Forward pass
                    outputs = self.forward(x_data)
                    assert not torch.isnan(outputs).any(), "NaN in outputs"
                    loss: torch.Tensor = loss_fn(outputs, y_data)
                    assert not torch.isnan(loss).any(), "NaN in loss"

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    self.clamp()

                    callbacks(
                        net=self,
                        epoch=epoch,
                        batch_id=i,
                        outputs=outputs,
                        loss=loss.item(),
                        call_type="train.batch.after",
                    )
                callbacks(net=self, epoch=epoch, call_type="train.epoch.after")
            callbacks(net=self, call_type="train.after")

        def test(
            self, dataloader: DataLoader, callbacks: CallbackTypes = "default"
        ) -> tuple[int, int]:
            callbacks = CallbackFactory.parse_callbacks(callbacks)
            self.eval()
            with torch.no_grad():
                callbacks(
                    net=self,
                    dataloader_length=dataloader.len("test"),
                    call_type="test.before",
                )
                correct = 0
                total = 0
                for i, (x_data, y_data) in enumerate(dataloader("test")):
                    callbacks(
                        net=self,
                        batch_id=i,
                        call_type="test.batch.before",
                    )

                    x_data = x_data.to(self.config.device)
                    y_data = y_data.to(self.config.device)

                    outputs = self(x_data)
                    # TODO: this should be a function
                    _, predicted = torch.max(outputs.data, 1)
                    _, expected = torch.max(y_data, 1)
                    total += y_data.size(0)
                    correct += (predicted == expected).sum().item()

                    callbacks(
                        net=self,
                        batch_id=i,
                        total_test_points=total,
                        correct_test_points=correct,
                        predicted=predicted,
                        expected=expected,
                        call_type="test.batch.after",
                    )
            callbacks(
                net=self,
                total_test_points=total,
                correct_test_points=correct,
                call_type="test.after",
            )
            return total, correct

        def __repr__(self):
            return f"{self.config.name}:\n{self.layers}"
