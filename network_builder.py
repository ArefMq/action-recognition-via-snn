from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LayerInterface:
    # This is the number of neurons on dense layer and the number of channels on convolutional layer
    out_features: Optional[int] = None
    out_shape: Optional[tuple] = None
    name: str | None = None

    @property
    def is_conv(self):
        return self.out_shape is not None or self.out_shape == [1]

    def __str__(self) -> str:
        shape = ""
        if self.is_conv:
            shape = "~  " + "x".join([str(f) for f in self.out_shape])
        return f"{self.name} ({self.out_features}{shape})"


@dataclass
class UncompiledLayer(LayerInterface):
    module: str = "?"  # e.g. 'torch.nn.Linear'

    last_layer: Optional[LayerInterface] = None

    def __post_init__(self):
        if self.name is None:
            self.name = self.module

    @property
    def in_features(self):
        return self.last_layer.out_features

    @property
    def in_shape(self):
        return self.last_layer.out_shape

    def compile(self):
        shape = ""
        if self.is_conv:
            shape = "~  " + "x".join([str(f) for f in self.out_shape])
        text = f"{self.in_features} -> {self.out_features}"

        out_params = np.prod(self.out_shape) * self.out_features
        in_params = np.prod(self.in_shape) * self.in_features
        return f"    {self.module} ({text}) \t:::{in_params} * {out_params}"

    def __str__(self):
        return self.compile()


class NetworkBuilder:
    def __init__(self):
        self.layers: list[LayerInterface | UncompiledLayer] = []

    def add_layer(
        self, module: str, out_features: Optional[int] = None, **kwargs
    ) -> "NetworkBuilder":
        if not self.layers:
            self.layers.append(LayerInterface(name="data_layer"))
        last_layer = self.layers[-1]
        self.layers.append(
            UncompiledLayer(
                module=module,
                out_features=out_features,
                last_layer=last_layer,
                **kwargs,
            )
        )
        return self

    def build(self, in_features: int | None = None, in_shape: tuple | None = None):
        data_layer = self.layers[0]
        if in_features is not None:
            data_layer.out_features = in_features
        if in_shape is not None:
            data_layer.out_shape = in_shape

        assert (
            data_layer.out_features is not None and data_layer.out_shape is not None
        ), "You must specify either in_features or in_shape"
        for layer in self.layers:
            print(str(layer))


if __name__ == "__main__":
    net = (
        NetworkBuilder()
        .add_layer("conv", 16)
        .add_layer("pooling")
        .add_layer("conv", 32)
        .add_layer("pooling")
        .add_layer("dense", 1500)
    )
    net.build(in_shape=(28, 28), in_features=1)
    # print("=" * 90, "\n")
    # net = (
    #     NetworkBuilder()
    #     .add_layer("conv", 16)
    #     .add_layer("pooling")
    #     .add_layer("conv", 32)
    #     .add_layer("pooling")
    #     .add_layer("dense", 1500)
    #     .add_layer("relu")
    #     .add_layer("dense", 10)
    # )
    # net.build()
