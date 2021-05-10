import torch
import numpy as np
import matplotlib.pyplot as plt

from scnn.Spike.spiking_neuron_base import SpikingNeuronBase


class ReadoutLayer(SpikingNeuronBase):
    IS_CONV = False
    IS_SPIKING = False
    HAS_PARAM = True

    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'ReadoutLayer'
        super(ReadoutLayer, self).__init__(*args, **kwargs)

        time_reduction = kwargs.get('time_reduction')
        if time_reduction == 'avg':
            time_reduction = 'mean'
        self.time_reduction = time_reduction

        self.w = torch.nn.Parameter(torch.empty((self.input_shape, self.output_shape)), requires_grad=True)
        if time_reduction == "max":
            self.beta = torch.nn.Parameter(torch.tensor(0.7 * np.ones((1))), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(self.output_shape), requires_grad=True)

    def trainable(self):
        if self.time_reduction == "max":
            return [self.w, self.b, self.beta]
        else:
            return [self.w, self.b]

    def __str__(self):
        return self.time_reduction + '(' + str(self.output_shape) + ')'

    def forward_function(self, x):
        batch_size = x.shape[0]
        h = torch.einsum("abc,cd->abd", x, self.w)
        norm = (self.w ** 2).sum(0)

        output = None
        if self.time_reduction == "max":
            nb_steps = x.shape[1]

            mem = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)
            mem_rec = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype, device=x.device)

            for t in range(nb_steps):
                # membrane potential update
                mem = mem * self.beta + (1 - self.beta) * h[:, t, :]
                mem_rec[:, t, :] = mem

            output = torch.max(mem_rec, 1)[0] / (norm + 1e-8) - self.b

        elif self.time_reduction == "mean":
            mem_rec = h
            output = torch.mean(mem_rec, 1) / (norm + 1e-8) - self.b

        # save mem_rec for plotting
        self.mem_rec_hist = mem_rec.detach().cpu().numpy()
        return output

    def reset_parameters(self):
        torch.nn.init.normal_(self.w, mean=self.w_init_mean,
                              std=self.w_init_std * np.sqrt(1. / np.prod(self.input_shape)))
        if self.time_reduction == "max":
            torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def clamp(self):
        if self.time_reduction == "max":
            self.beta.data.clamp_(0., 1.)

    def draw(self, batch_id=0, layer_id=None):
        mem_rec_hist = self.mem_rec_hist[batch_id]
        for i in range(mem_rec_hist.shape[1]):
            plt.plot(mem_rec_hist[:, i], label='mem')
            if i > 30:
                break
        if layer_id is not None:
            plt.title('layer: %s' % layer_id)
        plt.xlabel('Time')
        plt.ylabel('Membrace Potential')
        plt.show()

