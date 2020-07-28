import torch
import numpy as np
import matplotlib.pyplot as plt

from scnn.Spike.spiking_neuron_base import SpikingNeuronBase
from scnn.default_configs import *


class SpikingDenseLayer(SpikingNeuronBase):
    IS_CONV = False
    IS_SPIKING = True
    HAS_PARAM = True

    def __init__(self, *args, **kwargs):
        super(SpikingDenseLayer, self).__init__(*args, **kwargs)

        self.w = torch.nn.Parameter(torch.empty((self.input_shape, self.output_shape)), requires_grad=True)
        if self.recurrent:
            self.v = torch.nn.Parameter(torch.empty((self.output_shape, self.output_shape)), requires_grad=True)

        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(self.output_shape), requires_grad=True)

    def trainable(self):
        if self.recurrent:
            return [self.w, self.v, self.b, self.beta]
        else:
            return [self.w, self.b, self.beta]

    def __str__(self):
        return 'D(' + str(self.output_shape) + ('r' if self.recurrent else '') + ')'

    def forward_function(self, x):
        batch_size = x.shape[0]
        nb_steps = x.shape[1]

        h = torch.einsum("abc,cd->abd", x, self.w)

        # membrane potential
        mem = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)
        spk = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)

        # output spikes recording
        spk_rec = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype, device=x.device)
        self.mem_rec_hist = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype)

        if self.lateral_connections:
            d = torch.einsum("ab, ac -> bc", self.w, self.w)

        norm = (self.w ** 2).sum(0)

        for t in range(nb_steps):
            # reset term
            if self.lateral_connections:
                rst = torch.einsum("ab,bc ->ac", spk, d)
            else:
                rst = spk * self.b * norm

            input_ = h[:, t, :]
            if self.recurrent:
                input_ = input_ + torch.einsum("ab,bc->ac", spk, self.v)

            mem = (mem - rst) * self.beta + input_ * (1. - self.beta)
            mthr = torch.einsum("ab,b->ab", mem, 1. / (norm + EPSILON)) - self.b
            spk = self.spike_fn(mthr)

            spk_rec[:, t, :] = spk
            self.mem_rec_hist[:, t, :] = mem.detach().cpu()

            # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()
        self.mem_rec_hist = self.mem_rec_hist.numpy()

        return spk_rec

    def reset_parameters(self):
        torch.nn.init.normal_(self.w, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.input_shape))
        if self.recurrent:
            torch.nn.init.normal_(self.v, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.output_shape))
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def clamp(self):
        self.beta.data.clamp_(0., 1.)
        self.b.data.clamp_(min=0.)

    def draw(self, batch_id=0):
        mem_rec_hist = self.mem_rec_hist[batch_id]
        for i in range(mem_rec_hist.shape[1]):
            plt.plot(mem_rec_hist[:, i], label='mem')
            if i > 30:
                break
        plt.xlabel('Time')
        plt.ylabel('Membrace Potential')
        plt.show()

        spk_rec_hist = self.spk_rec_hist[batch_id]
        plt.plot(spk_rec_hist, 'b.')
        plt.xlabel('Time')
        plt.ylabel('Spikes')
        plt.show()

        plt.matshow(spk_rec_hist, origin="upper", aspect='auto')
        plt.xlabel('Neuron')
        plt.ylabel('Spike Time')
        plt.axis([-1, spk_rec_hist.shape[1], -1, spk_rec_hist.shape[0]])
        plt.show()
