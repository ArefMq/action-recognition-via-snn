from collections.abc import Callable

import torch
from torch import Tensor

from spikenet.tools.window import Window3D

PoolingReductionFunction = Callable[[Tensor, Window3D, Window3D], Tensor]


def max_spike_rate(spk_rec: Tensor, kernel: Window3D, stride: Window3D) -> Tensor:
    return torch.nn.functional.max_pool3d(
        spk_rec,
        kernel_size=kernel,
        stride=stride,
    )


def avg_spike_rate(spk_rec: Tensor, kernel: Window3D, stride: Window3D) -> Tensor:
    return torch.nn.functional.avg_pool3d(
        spk_rec,
        kernel_size=kernel,
        stride=stride,
    )


def spike_time(spk_rec: Tensor, kernel: Window3D, stride: Window3D) -> Tensor:
    raise NotImplementedError()
