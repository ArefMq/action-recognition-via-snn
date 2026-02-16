import math
from collections.abc import Iterator

import torch
from loguru import logger
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

    def __iadd__(self, other: "torch.nn.Module | Network") -> Self:
        if isinstance(other, Network):
            for layer in other._layers:
                self.add_layer(layer)
        else:
            self.add_layer(other)
        return self

    def __add__(self, other: "Network") -> "Network":
        new_net = Network(name=f"{self.name}+{other.name}")
        for layer in self._layers:
            new_net.add_layer(layer)
        for layer in other._layers:
            new_net.add_layer(layer)
        return new_net

    def fit(self, dataloader: DataLoader, **kwargs) -> Self:
        input_shape, output_shape = dataloader.shape
        self.compiled(input_features=input_shape[1], output_features=output_shape[1])
        assert self.validate_layers(), "Network layers are not properly configured"
        self.train(dataloader, **kwargs)
        self.test(dataloader)
        return self

    def test(self, dataloader: DataLoader) -> None: ...

    # ~~~~~~~~~~~~~~~~~~~~~~~ Compile Network ~~~~~~~~~~~~~~~~~~~~~~~~~
    def compiled(
        self,
        *args,
        input_features: int | None = None,
        output_features: int | None = None,
        **kwargs,
    ) -> None:
        if self._is_compiled:
            return
        self._populate_features(input_features, output_features)
        super().compile(*args, **kwargs)
        self._is_compiled = True

    def _populate_features(
        self,
        input_features: int | None = None,
        output_features: int | None = None,
    ):
        if self._layers[0].in_features is None:
            self._layers[0].in_features = input_features
        elif input_features is not None:
            logger.warning(
                "Input features provided but first layer already has input features, This parameter will be ignored."
            )
        if self._layers[-1].out_features is None:
            self._layers[-1].out_features = output_features
        elif output_features is not None:
            logger.warning(
                "Output features provided but last layer already has output features; This parameter will be ignored."
            )
        for i, layer in enumerate(self._layers):
            if not isinstance(layer, NeuronBase):
                continue
            if i == 0:
                continue

            if layer.in_features is None:
                layer.in_features = self._layers[i - 1].out_features
            if layer.out_features is None and i != len(self._layers) - 1:
                layer.out_features = layer.in_features

    def validate_layers(self) -> bool:
        return all(not (layer.in_features is None or layer.out_features is None) for layer in self._layers)

    @property
    def is_compiled(self) -> bool:
        return self._is_compiled and self.validate_layers()

    # ~~~~~~~~~~~~~~~~~~~~~~~~ Using Network ~~~~~~~~~~~~~~~~~~~~~~~~~~
    def forward(self, x: Tensor) -> Tensor:
        assert self.is_compiled, "Network must be compiled before forward pass"
        for layer in self._layers:
            x = layer(x)
        return x

    # ~~~~~~~~~~~~~~~~~~~~~~~ Utility Functions ~~~~~~~~~~~~~~~~~~~~~~~
    def initialize_parameters(self) -> None:
        assert self.is_compiled, "Network must be compiled before initializing parameters"
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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Visualisation ~~~~~~~~~~~~~~~~~~~~~~~~
    def summarise(self) -> None:
        """Print Summary of the network"""
        print(self.summary())

    def summary(self) -> str:
        """Return the summary of the network structure as string"""
        self.compiled()
        result = [
            self._layer_summary_other_layers(layer)
            if not isinstance(layer, NeuronBase)
            else self._layer_summary_spikenet_layer(layer)
            for layer in self._layers
        ]
        result_magnitude = math.ceil(math.log10(len(result))) if len(result) > 0 else 1
        return "\n".join(f"{i:0{result_magnitude}d}) {line}" for i, line in enumerate(result))

    def _layer_summary_spikenet_layer(self, layer: NeuronBase) -> str:
        name = type(layer).__name__
        in_f = str(layer.in_features) if layer.in_features is not None else "?"
        out_f = str(layer.out_features) if layer.out_features is not None else "?"
        shape = f"{in_f} \u2192 {out_f}"
        num_params = sum(p.numel() for p in layer.parameters())
        line = f"{name:<24s} {shape:<16s} {num_params:>8,} params"

        extras: list[str] = []
        if hasattr(layer, "kernel"):
            extras.append(f"kernel={layer.kernel}")
        if hasattr(layer, "time_reduction_fn") and layer.time_reduction_fn.__name__ != "no_time_reduction":
            extras.append(f"reduction={layer.time_reduction_fn.__name__}")
        if extras:
            line += "   " + ", ".join(extras)
        return line

    def _layer_summary_other_layers(self, layer: torch.nn.Module) -> str:
        name = layer.__class__.__name__
        num_params = sum(p.numel() for p in layer.parameters())
        return f"{name:<24s} {'':16s} {num_params:>8,} params"
