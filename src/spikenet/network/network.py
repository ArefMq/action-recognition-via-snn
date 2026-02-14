from collections.abc import Iterator

import torch
from torch import Tensor
from typing_extensions import Self

from spikenet.data import DataLoader
from spikenet.layers.neuron_base import NeuronBase
from spikenet.network.criterion import Criterion


class Network(torch.nn.Module):
    def __init__(
        self,
        *args,
        name: str = "SpikingNetwork",
        device: torch.device | None = None,
        optimizer_generator: torch.optim.Optimizer = torch.optim.SGD,
        optimizer: torch.optim.Optimizer | None = None,
        loss_fn: torch.nn.Module = torch.nn.NLLLoss,
        encoding: torch.nn.Module = None,
        epochs: int = 10,
        learning_rate: float = 0.0001,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.criterion = Criterion(
            optimizer_generator=optimizer_generator,
            optimizer=optimizer,
            loss_fn=loss_fn,
            encoding=encoding or torch.nn.LogSoftmax(dim=1),
            epochs=epochs,
            learning_rate=learning_rate,
        )
        self.name = name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._is_compiled: bool = False
        self._layers: list[torch.nn.Module] = []

    # ~~~~~~~~~~~~~~~~~~~~~ Constructing Network ~~~~~~~~~~~~~~~~~~~~~
    def add_layer(
        self,
        module: torch.nn.Module,
    ) -> Self:
        self._is_compiled = False
        self._layers.append(module)
        return self

    def fit(self, dataloader: DataLoader, **kwargs) -> Self:
        if not self._is_compiled:
            self.compile()
        self.train(dataloader, **kwargs)
        self.test(dataloader)
        return self

    # ~~~~~~~~~~~~~~~~~~~~~~~~ Using Network ~~~~~~~~~~~~~~~~~~~~~~~~~~
    def forward(self, x: Tensor) -> Tensor:
        for layer in self._layers:
            x = layer(x)
        return x

    # ~~~~~~~~~~~~~~~~~~~~~~~ Utility Functions ~~~~~~~~~~~~~~~~~~~~~~~
    def initialize_parameters(self) -> None:
        for layer in self._layers:
            if isinstance(layer, NeuronBase):
                layer.initialize_parameters()

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        for layer in self._layers:
            yield from layer.parameters()

    def clamp(self) -> None:
        for layer in self._layers:
            if isinstance(layer, NeuronBase):
                layer.clamp()
