import numpy as np

from scnn.Spike.spiking_neuron_base import SpikingNeuronBase


class ReadInLayer(SpikingNeuronBase):
    IS_CONV = True
    IS_SPIKING = False
    HAS_PARAM = False

    def __init__(self, *args, **kwargs):
        super(ReadInLayer, self).__init__(*args, **kwargs)

        if self.output_shape is None:
            self.output_shape = self.input_shape
        if self.output_channels:
            self.output_channels = self.input_channels

    def __str__(self):
        return 'I(' + str(self.input_channels) + 'x' + 'x'.join([str(i) for i in self.input_shape]) + ')'

    def forward_function(self, x):
        batch_size = x.shape[0]
        nb_steps = x.shape[1]

        if self.flatten_output:
            x = x.view(batch_size, nb_steps, self.output_channels * np.prod(self.output_shape))
        else:
            x = x.view(batch_size, self.output_channels, nb_steps, *self.output_shape)

        self.spk_rec_hist = x.detach().cpu().numpy()
        return x
