import torch
import numpy as np

from scnn.Spike.spiking_neuron_base import SpikingNeuronBase


class SpikingPool2DLayer(SpikingNeuronBase):
    IS_CONV = True
    IS_SPIKING = False
    HAS_PARAM = False

    def __init__(self, *args, **kwargs):
        super(SpikingPool2DLayer, self).__init__(*args, **kwargs)

        if self.output_channels is None:
            self.output_channels = self.input_channels

        if self.stride is None:
            self.stride = self.kernel_size

        if self.output_shape is None:
            self.output_shape = [int(1+(i-k)/s) for i, k, s in zip(self.input_shape, self.kernel_size, self.stride)]

    def __str__(self):
        # FIXME: handle other variations
        return 'P(' + str(self.kernel_size[0]) + ')'

    def forward_function(self, x):
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
