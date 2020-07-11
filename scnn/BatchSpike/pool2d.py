import torch
import numpy as np

from scnn.default_configs import *


class SpikingPool2DLayer(torch.nn.Module):
    IS_CONV = True
    IS_SPIKING = False
    HAS_PARAM = False

    def __init__(self, input_shape, input_channels, kernel_size=(2, 2), stride=None,
                 output_shape=None, output_channels=None, flatten_output=False):

        super(SpikingPool2DLayer, self).__init__()

        if output_channels is None:
            output_channels = input_channels
        if stride is None:
            stride = kernel_size

        if output_shape is None:
            output_shape = [int(1+(i-k)/s) for i, k, s in zip(input_shape, kernel_size, stride)]

        self.kernel_size = np.array(kernel_size)
        self.stride = np.array(stride)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.flatten_output = flatten_output
        self.spk_rec_hist = None

    def get_trainable_parameters(self, lr=None, weight_decay=None):
        return []

    def serialize(self):
        return {
            'type': 'pool2d',
            'params': {
                'kernel_size': self.kernel_size,
                'stride': self.stride
            }
        }

    def serialize_to_text(self):
        # FIXME: handle other variations
        return 'P(' + str(self.kernel_size[0]) + ')'

    def forward(self, x):
        batch_size = x.shape[0]
        nb_steps = x.shape[2]
        spk_rec = torch.zeros((batch_size, self.output_channels, nb_steps, *self.output_shape), dtype=x.dtype, device=x.device)

        for t in range(nb_steps):
            pool_x_t = torch.nn.functional.max_pool2d(x[:, :, t, :, :], kernel_size=tuple(self.kernel_size), stride=tuple(self.stride))
            spk_rec[:, :, t, :, :] = pool_x_t

        self.spk_rec_hist = spk_rec.detach().cpu().numpy()

        if self.flatten_output:
            output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = output.view(batch_size, nb_steps, self.output_channels * np.prod(self.output_shape))
        else:
            output = spk_rec

        return output

    def clamp(self):
        pass

    def draw(self, *kwargs):
        pass
