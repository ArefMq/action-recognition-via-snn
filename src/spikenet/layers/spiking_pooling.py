from torch import Tensor

from spikenet.functions.pooling_reduction import PoolingReductionFunction, max_spike_rate
from spikenet.layers.spiking_base import SpikingNeuron
from spikenet.tools.window import window_to_and_array


class SpikingPoolingLayer(SpikingNeuron):
    """
    Spiking Pooling Layer: This layer is used to create a pooling layer of spiking neurons.

    Args:
        name (str): Name of the layer (default: "SpikingPoolingLayer")
        in_features (int): Number of input features. None for getting the value from previous layer or input tensor.
        out_features (int): Number of output features.
        w_init_mean (float): Mean of the normal distribution used to initialize the weights.
        w_init_std (float): Standard deviation of the normal distribution used to initialize the weights.
        spike_fn (Callable): The spike function to use (default: SurrogateHeaviside.apply)
        time_reduction (str or TimeReduction): The time reduction method to use (default: No time reduction)
        beta_init_std (float): Standard deviation of the normal distribution used to initialize the beta parameter.
        beta_init_mean (float): Mean of the normal distribution used to initialize the beta parameter.
        b_init_std (float): Standard deviation of the normal distribution used to initialize the b parameter.
        b_init_mean (float): Mean of the normal distribution used to initialize the b parameter.
        stride (int or np.ndarray): The stride of the pooling operation (default: [1, 2, 2])
        kernel (int or np.ndarray): The size of the pooling kernel (default: [1, 2, 2])
        reduction (PollingReduction): The reduction method to use (default: max_spike_rate)

    NOTE: Not to confuse reduction with time_reduction!
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(name=kwargs.pop("name", "SpikingPoolingLayer"), **kwargs)
        self.stride = window_to_and_array(kwargs.get("stride", (1, 2, 2)))
        self.kernel = window_to_and_array(kwargs.get("kernel", (1, 2, 2)))
        self.reduction: PoolingReductionFunction = kwargs.get("reduction", max_spike_rate)

    def spike_forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        return self.reduction(x, self.kernel, self.stride), None
