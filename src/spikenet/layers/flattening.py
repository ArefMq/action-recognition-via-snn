import torch
from torch import Tensor

from spikenet.layers.neuron_base import NeuronBase


class Flatten(NeuronBase):
    """
    Flatten layer: flattens the input tensor to a 2D tensor
    This is used after the convolutional layers to flatten the output tensor to be
    used by the dense layer.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(out_features=None, **kwargs)

    def forward(self, spk_rec: Tensor) -> Tensor:
        """
        Forward pass of the flatten layer.

        Args:
            spk_rec (Tensor): The spike record tensor with shape (batch_size, channels, nb_steps, *spatial_dims)

        Returns:
            Tensor: The flattened tensor with shape (batch_size, nb_steps, flattened_features)
        """
        (batch_size, nb_channels, nb_steps, *_) = spk_rec.shape
        output = torch.transpose(spk_rec, 1, 2).contiguous()
        result = output.view(batch_size, nb_steps, -1)
        self.out_features = result.shape[2]
        return result
