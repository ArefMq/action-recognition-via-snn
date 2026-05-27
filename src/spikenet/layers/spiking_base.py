from abc import ABC, abstractmethod
from collections.abc import Callable

from torch import Tensor

from spikenet.functions.heaviside import SurrogateHeaviside
from spikenet.functions.time_reduction import TimeReductionFunction, no_time_reduction
from spikenet.layers.neuron_base import NeuronBase
from spikenet.visual.mixins import LayerPlottable


class SpikingNeuron(NeuronBase, LayerPlottable, ABC):
    """
    Base class for all spiking neuron layers used in the SpikeNet framework.

    Args:
        name (str): Name of the layer (default: "SpikingNeuron")
        in_features (int | None): Number of input features. None means defer specification to compile time.
        out_features (int | None): Number of output features. None means defer specification to compile time.
        w_init_mean (float): Mean of the normal distribution used to initialize the weights.
        w_init_std (float): Standard deviation of the normal distribution used to initialize the weights.
        spike_fn (Callable): The spike function to use (default: SurrogateHeaviside.apply)
        time_reduction (Callable | None): The time reduction method to use (default: None)
        beta_init_std (float): Standard deviation of the normal distribution used to initialize the beta parameter.
        beta_init_mean (float): Mean of the normal distribution used to initialize the beta parameter.
        b_init_std (float): Standard deviation of the normal distribution used to initialize the b parameter.
        b_init_mean (float): Mean of the normal distribution used to initialize the b parameter.
    """

    def __init__(self, out_features: int | None = None, **kwargs) -> None:
        super().__init__(out_features=out_features, name=kwargs.pop("name", "SpikingNeuron"), **kwargs)
        self.spike_fn: Callable = kwargs.get("spike_fn", SurrogateHeaviside.apply)
        self.time_reduction_fn: TimeReductionFunction = kwargs.get("time_reduction", no_time_reduction)

        # Internal attributes .....................
        self._mem_rec: Tensor | None = None
        self._spike_rec: Tensor | None = None

        # Register learnable parameters as None — subclasses populate
        # these in initialize_parameters(). Using register_parameter
        # keeps them compatible with PyTorch's Module parameter system.
        self.register_parameter("w", None)
        self.register_parameter("beta", None)
        self.register_parameter("b", None)

        # Initialization parameters ..............
        self.beta_init_std: float = kwargs.get("beta_init_std", 0.01)
        self.beta_init_mean: float = kwargs.get("beta_init_mean", 0.7)
        self.b_init_std: float = kwargs.get("b_init_std", 0.01)
        self.b_init_mean: float = kwargs.get("b_init_mean", 1.0)

    @property
    def mem_rec(self) -> Tensor:
        """Membrane potential record of the neuron.

        Returns:
            a tensor of shape (batch_size, time_steps, *out_features)

        Raises:
            AssertionError: if mem_rec is not initialized
        """
        assert self._mem_rec is not None, "mem_rec is not initialized"
        return self._mem_rec

    @property
    def spike_rec(self) -> Tensor:
        """Spike record of the neuron.

        Returns:
            a binary tensor of shape (batch_size, time_steps, *out_features)

        Raises:
            AssertionError: if spike_rec is not initialized
        """
        assert self._spike_rec is not None, "spike_rec is not initialized"
        return self._spike_rec

    @property
    def w_norm(self) -> Tensor:
        """Weight norm of the neuron.

        Returns:
            a tensor of shape (out_features,)

        Raises:
            AssertionError: if w is not initialized
            AssertionError: if w_norm contains NaN values
        """
        assert self.w is not None, "Parameters w are not initialized"
        norm = (self.w**2).sum(0)
        self._check_nan(norm, "w_norm")
        return norm

    def clamp(self) -> None:
        """Implementation of the parameter clamping for the neuron."""
        if self.beta is not None:
            self.beta.data.clamp_(0.0, 1.0)
        if self.b is not None:
            self.b.data.clamp_(min=0.0)

    def forward(self, x: Tensor) -> Tensor:
        """The forward pass of the neuron.

        Args:
            x (Tensor): the input tensor value

        Returns:
            Tensor: the output tensor
        """
        self._assert(x.any(), "No input spikes.", warning=True)
        spk_rec, mem_rec = self.spike_forward(x)
        if mem_rec is not None:
            self._check_nan(mem_rec, "mem_rec")
        self._check_nan(spk_rec, "spk_rec")

        self._mem_rec = mem_rec
        reduced_spike_rec = self.time_reduction_fn(self, spk_rec, mem_rec)
        self._spike_rec = reduced_spike_rec
        self._check_nan(reduced_spike_rec, "output")
        return reduced_spike_rec

    @abstractmethod
    def spike_forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        """Spiking specific implementation of the forward pass.

        Args:
            x (Tensor): the input tensor

        Returns:
            tuple[Tensor, Tensor | None]: the spikes and the membrane potential
        """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HELPERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def avg_mem(self) -> float:
        """Return the average membrane potential. Used to debug layer activity."""
        assert self._mem_rec is not None, "mem_rec is not initialized"
        return self._mem_rec.mean().item()

    def avg_spike(self) -> float:
        """Return the average spike count. Used to debug layer activity."""
        assert self._spike_rec is not None, "spike_rec is not initialized"
        return self._spike_rec.mean().item()

    def mem_percentage(self) -> float:
        """Return the percentage of membrane potential. Used to debug layer activity."""
        assert self._mem_rec is not None, "mem_rec is not initialized"
        return (self._mem_rec > 0).sum().item() / self._mem_rec.numel()

    def spike_percentage(self) -> float:
        """Return the percentage of spike count. Used to debug layer activity."""
        assert self._spike_rec is not None, "spike_rec is not initialized"
        return (self._spike_rec > 0).sum().item() / self._spike_rec.numel()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ STRING REPRESENTATION ~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __repr__(self) -> str:
        return (
            f"{self.name}(in={self.in_features}, out={self.out_features}, "
            f"mean={self.w_init_mean}, std={self.w_init_std}, "
            f"spike_fn={self.spike_fn.__name__}"
            f", time_reduction={self.time_reduction_fn.__name__})"
        )

    def __str__(self) -> str:
        return f"{self.name}({self.out_features}, spiking)"
