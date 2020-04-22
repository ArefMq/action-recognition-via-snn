import torch
import numpy as np


class ReadInLayer(torch.nn.Module):
    IS_CONV = True
    IS_SPIKING = False
    HAS_PARAM = False

    def __init__(self, input_shape, input_channels=1, output_shape=None, output_channels=None, flatten_output=False):
        super(ReadInLayer, self).__init__()

        self.input_shape = input_shape
        self.input_channels = input_channels

        self.output_shape = output_shape if output_shape is not None else input_shape
        self.output_channels = output_channels if output_channels is not None else input_channels

        self.flatten_output = flatten_output

    def get_trainable_parameters(self, lr):
        return []

    def forward(self, x):
        batch_size = x.shape[0]
        nb_steps = x.shape[1]

        if self.flatten_output:
            return x.view(batch_size, nb_steps, self.output_channels * np.prod(self.output_shape))
        else:
            return x.view(batch_size, self.output_channels, nb_steps, *self.output_shape)

    def reset_parameters(self):
        pass

    def clamp(self):
        pass

