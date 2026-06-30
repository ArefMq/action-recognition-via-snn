from collections.abc import Callable

import torch
from torch import Tensor

from spikenet.tools.window import Window3D

PoolingReductionFunction = Callable[[Tensor, Window3D, Window3D], Tensor]


def max_spike_rate(spk_rec: Tensor, kernel: Window3D, stride: Window3D) -> Tensor:
    """Max pooling over spike recordings; in other words, the neuron with the highest spike rate in each window will
    be spiked.

    Args:
        spk_rec: Spike recordings with shape (batch, channels, time, height, width)
        kernel: Kernel size for pooling
        stride: Stride for pooling

    Returns:
        Pooled spike recordings with shape (batch, channels, time//stride[0], height//stride[1], width//stride[2])
    """
    return torch.nn.functional.max_pool3d(
        spk_rec,
        kernel_size=kernel,
        stride=stride,
    )


def avg_spike_rate(spk_rec: Tensor, kernel: Window3D, stride: Window3D) -> Tensor:
    """Average pooling over spike recordings; in other words, the average spike rate in each window will be used.

    Args:
        spk_rec: Spike recordings with shape (batch, channels, time, height, width)
        kernel: Kernel size for pooling
        stride: Stride for pooling

    Returns:
        Pooled spike recordings with shape (batch, channels, time//stride[0], height//stride[1], width//stride[2])
    """
    return torch.nn.functional.avg_pool3d(
        spk_rec,
        kernel_size=kernel,
        stride=stride,
    )


def spike_time(spk_rec: Tensor, kernel: Window3D, stride: Window3D) -> Tensor:
    """First spike time pooling over spike recordings; in other words the neuron that fires first in each window will
    be spiked.

    Args:
        spk_rec: Spike recordings with shape (batch, channels, time, height, width)
        kernel: Kernel size for pooling
        stride: Stride for pooling

    Returns:
        Pooled spike recordings with shape (batch, channels, time//stride[0], height//stride[1], width//stride[2])
    """
    raise NotImplementedError()
