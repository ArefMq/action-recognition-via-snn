import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scnn.Spike.dense import SpikingDenseLayer
from scnn.default_configs import *


class SpikingDenseStream(SpikingDenseLayer):
    def __init__(self, *args, **kwargs):
        super(SpikingDenseStream, self).__init__(*args, **kwargs)
        self.histogram_memory_size = kwargs.get('histogram_memory_size', 50)

        # Variables
        self.mem = None
        self.spk = None
        self.history_counter = 0

    def __str__(self):
        return 'D.St(' + str(self.output_shape) + ('r' if self.recurrent else '') + ')'

    def forward_function(self, x):
        batch_size = x.shape[0]
        if self.mem is None:
            self.reset_mem(batch_size, x.device, x.dtype)

        h = torch.einsum("ac,cd->ad", x, self.w)

        if self.lateral_connections:
            d = torch.einsum("ab, ac -> bc", self.w, self.w)

        norm = (self.w ** 2).sum(0)

        # reset term
        if self.lateral_connections:
            rst = torch.einsum("ab,bc ->ac", self.spk, d)
        else:
            rst = self.spk * self.b * norm

        if self.recurrent:
            h = h + torch.einsum("ab,bc->ac", self.spk, self.v)

        self.mem = (self.mem - rst) * self.beta + h * (1. - self.beta)
        mthr = torch.einsum("ab,b->ab", self.mem, 1. / (norm + EPSILON)) - self.b
        self.spk = self.spike_fn(mthr)

        self.spk_rec_hist[:, self.history_counter, :] = self.spk.detach().cpu()
        self.mem_rec_hist[:, self.history_counter, :] = self.mem.detach().cpu()
        self.history_counter += 1
        if self.history_counter >= self.histogram_memory_size:
            self.history_counter = 0

        return self.spk

    def reset_mem(self, batch_size, x_device, x_dtype):
        self.mem = torch.zeros((batch_size, self.output_shape), dtype=x_dtype, device=x_device)
        self.spk = torch.zeros((batch_size, self.output_shape), dtype=x_dtype, device=x_device)
        self.spk_rec_hist = torch.zeros((batch_size, self.histogram_memory_size, self.output_shape), dtype=x_dtype)
        self.mem_rec_hist = torch.zeros((batch_size, self.histogram_memory_size, self.output_shape), dtype=x_dtype)
        self.history_counter = 0

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

        plt.matshow(spk_rec_hist)
        plt.xlabel('Neuron')
        plt.ylabel('Spike Time')
        plt.axis([-1, spk_rec_hist.shape[1], -1, spk_rec_hist.shape[0]])
        plt.show()
