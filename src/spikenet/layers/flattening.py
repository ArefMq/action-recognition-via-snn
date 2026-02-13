import torch
from spikenet.layers.neuron_base import NeuronBase


class Flatten(NeuronBase):
    """
    Flatten layer: flattens the input tensor to a 2D tensor
    This is used after the convolutional layers to flatten the output tensor to be
    used by the dense layer.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (batch_size, _, nb_steps, *_) = x.shape
        output = torch.transpose(x, 1, 2).contiguous()
        output = output.view(batch_size, nb_steps, self.out_features)
        return output

    def plot_mem(*args, **kwargs) -> None:
        # Flatten layer does not have membrane potential
        pass

    def plot_spk(*args, **kwargs) -> None:
        # Flatten layer does not have spikes
        pass
        