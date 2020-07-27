import torch
import numpy as np

from scnn.Spike.conv1d import SpikingConv1DLayer


class SpikingConv1DStream(SpikingConv1DLayer):
    def __init__(self, *args, **kwargs):
        super(SpikingConv1DStream, self).__init__(*args, **kwargs)
        self.histogram_memory_size = kwargs.get('histogram_memory_size', 50)

        # Variables
        self.mem = None
        self.spk = None
        self.history_counter = 0

    def forward_function(self, x):
        raise NotImplementedError()

    def reset_mem(self, batch_size, x_device, x_dtype):
        self.mem = torch.zeros((batch_size, self.output_channels, *self.output_shape), dtype=x_dtype, device=x_device)
        self.spk = torch.zeros((batch_size, self.output_channels, *self.output_shape), dtype=x_dtype, device=x_device)

        self.spk_rec_hist = torch.zeros(
            (batch_size, self.output_channels, self.histogram_memory_size, *self.output_shape), dtype=x_dtype)
        self.mem_rec_hist = torch.zeros(
            (batch_size, self.output_channels, self.histogram_memory_size, *self.output_shape), dtype=x_dtype)

        self.history_counter = 0
