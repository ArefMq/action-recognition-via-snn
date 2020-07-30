import torch
import numpy as np

from scnn.Spike.pool2d import SpikingPool2DLayer


class SpikingPool2DStream(SpikingPool2DLayer):
    def __init__(self, *args, **kwargs):
        super(SpikingPool2DStream, self).__init__(*args, **kwargs)
        self.histogram_memory_size = kwargs.get('histogram_memory_size', 50)
        self.history_counter = 0

    def __str__(self):
        # FIXME: handle other variations
        return 'P.St(' + str(self.kernel_size[0]) + ')'

    def forward_function(self, x):
        batch_size = x.shape[0]
        if self.spk_rec_hist is None:
            self.reset_mem(batch_size, x.device, x.dtype)

        pool_x_t = torch.nn.functional.max_pool2d(x[:, :, :, :], kernel_size=tuple(self.kernel_size),
                                                  stride=tuple(self.stride))

        self.spk_rec_hist[:, :, self.history_counter, :, :] = pool_x_t.detach().cpu()
        self.history_counter += 1
        if self.history_counter >= self.histogram_memory_size:
            self.history_counter = 0

        if self.flatten_output:
            #             output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = pool_x_t.view(batch_size, self.output_channels * np.prod(self.output_shape))
        else:
            output = pool_x_t

        return output

    def reset_mem(self, batch_size, x_device, x_dtype):
        self.spk_rec_hist = torch.zeros(
            (batch_size, self.output_channels, self.histogram_memory_size, *self.output_shape), dtype=x_dtype)
        self.history_counter = 0
