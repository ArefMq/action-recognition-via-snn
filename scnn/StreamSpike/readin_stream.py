import torch
import numpy as np

from scnn.Spike.readin import ReadInLayer


class ReadInStream(ReadInLayer):
    def __init__(self, *args, **kwargs):
        super(ReadInStream, self).__init__(*args, **kwargs)
        self.histogram_memory_size = kwargs.get('histogram_memory_size', 50)
        self.history_counter = 0

    def serialize_to_text(self):
        return 'I.St(' + str(self.input_channels) + 'x' + 'x'.join([str(i) for i in self.input_shape]) + ')'

    def forward_function(self, x):
        batch_size = x.shape[0]
        self.history_counter += 1
        if self.history_counter >= self.histogram_memory_size:
            self.history_counter = 0

        if self.flatten_output:
            return x.view(batch_size, self.output_channels * np.prod(self.output_shape))
        else:
            print('a:', self.output_channels)
            print('b:', self.output_shape)
            print('x:', x.shape)
            return x.view(batch_size, self.output_channels, *self.output_shape)

    def reset_mem(self, batch_size, x_device, x_dtype):
        self.history_counter = 0


