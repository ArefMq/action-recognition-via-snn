from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Union
import numpy as np
import torch
from torch.nn.parameter import Parameter

from spikenet.dataloader import DataLoader
from spikenet.layers.flattening import Flatten
from spikenet.layers.spiking_pooling import SpikingPoolingLayer
from spikenet.tools.utils.callbacks_helper import CallbackFactory, CallbackTypes

# TODO: add summary function to both Network and CompiledNetwork
# TODO: upon creation network, the layer-counter should be reset.
# or even better, the layers should be a property that is generated in the network object
# TODO: add repr to the layers


@dataclass
class Network:
    @dataclass
    class _UncompiledLayer:
        module: torch.nn.Module
        last_layer: Optional["Network._UncompiledLayer"] = None
        out_features: int | None = None

        aditional_args: dict[str, Any] = field(default_factory=dict)

        def is_ready_to_compile(self) -> bool:
            if self.out_features is None:
                return True  # This is a function layer such as ReLU
            return self.get_in_features() is not None
        
        def get_in_size(self) -> tuple[int]:
            return self.aditional_args.get("in_size", None)

        def get_out_size(self) -> tuple[int]:
            out_size = self.aditional_args.get("out_size", None)
            if out_size is not None:
                return out_size
            return self.get_in_size()

        def get_in_features(self) -> int:
            if self.out_features is None:
                return None
            return (
                self.in_features
                if self.in_features is not None
                else self.borrowed_in_features
            )

        def get_out_features(self) -> int:
            return (
                self.out_features
                if self.out_features is not None
                else self.borrowed_in_features
            )

        def compile(self) -> torch.nn.Module:
            kwargs = self.aditional_args
            if (infe := self.get_in_features()) is not None:
                kwargs["in_features"] = infe
            if self.out_features is not None:
                kwargs["out_features"] = self.out_features
            return self.module(**kwargs)

        def __str__(self) -> str:
            if self.out_features is not None:
                text = f" ({self.get_in_features()} -> {self.out_features})"
                if self.get_in_size() is not None:
                    text += f" F[{self.get_in_size()} -> {self.get_out_size()}]"
            else:
                text = ""
            cnv = "  Conv" if self.is_conv else ""
            return f"{self.module.__name__}{text}{cnv}"
        
        def flatten_size(self) -> int:
            out_size = self.get_out_size()
            if out_size is None:
                return self.get_out_features()
            return int(np.prod(out_size) * self.get_out_features())
        
        @property
        def is_conv(self) -> bool:
            return self.IsConv(self.module)
        
        @staticmethod
        def IsConv(module: torch.nn.Module) -> bool:
            return hasattr(module, "is_conv") and module.is_conv
            

    @dataclass
    class Criterion:
        optimizer_generator: torch.optim.Optimizer = torch.optim.SGD
        optimizer: torch.optim.Optimizer | Any | None = None
        loss_fn: torch.nn.Module = torch.nn.NLLLoss
        encoding = torch.nn.LogSoftmax(dim=1)
        epochs: int = 10
        learning_rate: float = 0.0001

        def get_optim(self, net: torch.nn.Module) -> torch.optim.Optimizer:
            if self.optimizer is not None:
                return self.optimizer
            return self.optimizer_generator(
                net.parameters(), lr=self.learning_rate, momentum=0.9
            )

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

    name: str = "Network"
    layers: list[_UncompiledLayer] | torch.nn.ModuleList = field(default_factory=list)
    device: torch.device = field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    criterion: Criterion = field(default_factory=Criterion)
    __is_compiled: bool = False

    def add_layer(
        self,
        module: torch.nn.Module,
        out_features: int | None = None,
        in_features: int | None = None,
        **kwargs,
    ) -> "Network":
        """
        Add a layer to the network

        Args:
            module (torch.nn.Module): the layer to add
            out_features (int, optional): the number of output features Default is the output of the previous layer.s
            in_features (int, optional): the number of input features. Default is the output of the previous layer.
            **kwargs: additional arguments to pass to the layer
        """
        # FIXME: this definately needs a refactor
        self.__is_compiled = False
        borrowed_in_features = None
        if self.layers:
            last_layer = self.layers[-1]
            borrowed_in_features = self.layers[-1].get_out_features()

            if last_layer.is_conv and not self._UncompiledLayer.IsConv(module):
                borrowed_in_features = last_layer.flatten_size()
                self.layers.append(
                    self._UncompiledLayer(
                        module=Flatten, out_features=borrowed_in_features,
                        in_features=last_layer.get_out_features(),
                    )
                )
        
        if in_features is None and module == SpikingPoolingLayer:
            out_features = borrowed_in_features

        self.layers.append(
            self._UncompiledLayer(
                module=module,
                in_features=in_features,
                out_features=out_features,
                aditional_args=kwargs,
                borrowed_in_features=borrowed_in_features,
            )
        )
        return self
    
    def build(self) -> "Network._CompiledNetwork":
        compiled_layers = []
        for layer in self.layers:
            assert layer.is_ready_to_compile(), f"{layer} is not ready to compile."
            new_layer = layer.compile().to(self.device)
            compiled_layers.append(new_layer)
        self.layers = torch.nn.ModuleList(compiled_layers)
        self.__is_compiled = True
        return Network._CompiledNetwork(self).to(self.device)

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

    def fit_on(self, dataloader: DataLoader, **kwargs) -> "Network._CompiledNetwork":
        if self.layers[0].in_features is None:
            input_shape, _ = dataloader.shape
            # FIXME : this will not work with conv.layers
            self.layers[0].in_features = input_shape[-1]
        net = self.build()
        net.fit(dataloader, **kwargs)
        return net

    def summary(self, details: bool = False) -> None:
        assert not details or self.__is_compiled, "Uncompiled network cannot show details"
        print(f"Network: {self.name} [Uncompiled Network]")
        print("-" * 50)
        for i, layer in enumerate(self.layers):
            print(f"{i}) {layer}")
            if details:
                print(f"    - {layer.details()}\n")

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
                if hasattr(layer, "initialize_parameters"):
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
                if hasattr(layer, "clamp"):
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

                    # Forward pass
                    outputs = self.forward(x_data)
                    if self.config.criterion.encoding is not None:
                        outputs = self.config.criterion.encoding(outputs)
                    loss: torch.Tensor = loss_fn(outputs, y_data)

                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    self.clamp()
                    optimizer.zero_grad()

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
                    predicted = torch.max(outputs, 1).indices
                    if len(y_data.shape) > 1:
                        y_data = torch.max(y_data, 1).indices
                    total += y_data.size(0)
                    correct += (predicted == y_data).sum().item()

                    callbacks(
                        net=self,
                        batch_id=i,
                        total_test_points=total,
                        correct_test_points=correct,
                        predicted=predicted,
                        expected=y_data,
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

        def summary(self):
            print(f"Network: {self.config.name}")
            print("-" * 50)
            for i, layer in enumerate(self.layers):
                print(f"{i}) {layer}")

        # ~~~~~~~~ Plotting ~~~~~~~~
        def plot_activity(self):
            import matplotlib.pyplot as plt

            for i, lyr in enumerate(self.layers):
                fig, axs = plt.subplots(1, 2, figsize=(15, 5))
                lyr.plot_mem(ax=axs[0])
                lyr.plot_spk(ax=axs[1])
                fig.suptitle(f"Layer {i}")
