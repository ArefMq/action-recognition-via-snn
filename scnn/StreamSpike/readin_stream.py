import torch
import numpy as np

from scnn.default_configs import *


class ReadInStream(torch.nn.Module):
    IS_CONV = True
    IS_SPIKING = False
    HAS_PARAM = False

    def __init__(self, input_shape, input_channels=1, output_shape=None, output_channels=None,
                 flatten_output=False):
        super(ReadInStream, self).__init__()

        self.input_shape = input_shape
        self.input_channels = input_channels

        self.output_shape = output_shape if output_shape is not None else input_shape
        self.output_channels = output_channels if output_channels is not None else input_channels

        self.flatten_output = flatten_output
        self.history_counter = 0

    def get_trainable_parameters(self, lr=None, weight_decay=None):
        return []

    def serialize(self):
        return {
            'type': 'readin_stream',
            'params': {
                'input_shape': self.input_shape,
                'input_channels': self.input_channels
            }
        }

    def serialize_to_text(self):
        return 'I.St(' + str(self.input_channels) + 'x' + 'x'.join([str(i) for i in self.input_shape]) + ')'

    def forward(self, x):
        batch_size = x.shape[0]
        self.history_counter += 1
        if self.history_counter >= HISTOGRAM_MEMORY_SIZE:
            self.history_counter = 0

        if self.flatten_output:
            return x.view(batch_size, self.output_channels * np.prod(self.output_shape))
        else:
            return x.view(batch_size, self.output_channels, *self.output_shape)

    def reset_mem(self, batch_size, x_device, x_dtype):
        self.history_counter = 0

    def reset_parameters(self):
        pass

    def clamp(self):
        pass


