import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scnn.StreamSpike.conv1d_stream import SpikingConv1DStream
from scnn.default_configs import *


class SpikingConv2DStream(SpikingConv1DStream):
    def __init__(self, *args, **kwargs):
        super(SpikingConv2DStream, self).__init__(*args, **kwargs)

    def __str__(self):
        # FIXME: re-write this
        return 'C2.St(' + str(self.output_channels) \
               + ',k' + str(self.kernel_size[0]) \
               + (',l' if self.lateral_connections else '') \
               + (',r' if self.recurrent else '') \
               + ')'

    def forward_function(self, x):
        batch_size = x.shape[0]
        if self.mem is None or self.spk is None:
            self.reset_mem(batch_size, x.device, x.dtype)

        stride = tuple(self.stride)
        padding = tuple(np.ceil(((self.kernel_size - 1) * self.dilation) / 2).astype(int))
        conv_x = torch.nn.functional.conv2d(x, self.w, padding=padding,
                                            dilation=tuple(self.dilation),
                                            stride=stride)
        conv_x = conv_x[:, :, :self.output_shape[0], :self.output_shape[1]]

        if self.lateral_connections:
            d = torch.einsum("abcde, fbcde -> af", self.w, self.w)
        b = self.b.unsqueeze(1).unsqueeze(1).repeat((1, *self.output_shape))

        norm = (self.w ** 2).sum((1, 2, 3))

        if self.lateral_connections:
            rst = torch.einsum("abcd,be ->aecd", self.spk, d)
        else:
            rst = torch.einsum("abcd,b,b->abcd", self.spk, self.b, norm)

        if self.recurrent:
            conv_x = conv_x + torch.einsum("abcd,be->aecd", self.spk, self.v)

        self.mem = (self.mem - rst) * self.beta + conv_x * (1. - self.beta)
        mthr = torch.einsum("abcd,b->abcd", self.mem, 1. / (norm + EPSILON)) - b
        self.spk = self.spike_fn(mthr)

        self.spk_rec_hist[:, :, self.history_counter, :, :] = self.spk.detach().cpu()
        self.mem_rec_hist[:, :, self.history_counter, :, :] = self.mem.detach().cpu()
        self.history_counter += 1
        if self.history_counter >= self.histogram_memory_size:
            self.history_counter = 0

        if self.flatten_output:
            output = self.spk.view(batch_size, self.output_channels * np.prod(self.output_shape))
        else:
            output = self.spk

        return output

    def draw(self, batch_id=0):
        spk_rec_hist = self.spk_rec_hist[batch_id]
        mem_rec_hist = self.mem_rec_hist[batch_id]

        time_step = mem_rec_hist.shape[1]
        channels = mem_rec_hist.shape[0]
        rest_shape = mem_rec_hist.shape[2:]

        tmp_spk = np.zeros((time_step, channels, *rest_shape))
        tmp_mem = np.zeros((time_step, channels, *rest_shape))
        for i in range(time_step):
            tmp_spk[i, :, :, :] = spk_rec_hist[:, i, :, :]
            tmp_mem[i, :, :, :] = mem_rec_hist[:, i, :, :]
        spk_rec_hist = tmp_spk
        mem_rec_hist = tmp_mem

        flat_spk = np.reshape(spk_rec_hist, (time_step, channels * np.prod(mem_rec_hist.shape[2:])))
        flat_mem = np.reshape(mem_rec_hist, (time_step, channels * np.prod(mem_rec_hist.shape[2:])))

        # Plot Flats
        max_flats = 25
        if flat_mem.shape[1] > max_flats:
            inx = np.random.randint(flat_mem.shape[1], size=max_flats)
            flat_spk = flat_spk[:, inx]
            flat_mem = flat_mem[:, inx]

        for i in range(flat_mem.shape[1]):
            plt.plot(flat_mem[:, i], label='mem')
        plt.xlabel('Time')
        plt.ylabel('Membrace Potential')
        plt.show()

        plt.plot(flat_spk, '.')
        plt.xlabel('Time')
        plt.ylabel('Spikes')
        plt.show()

        plt.matshow(flat_spk, cmap=plt.cm.gray_r, origin="upper", aspect='auto')
        plt.xlabel('Neuron')
        plt.ylabel('Spike Time')
        plt.axis([-1, flat_spk.shape[1], -1, flat_spk.shape[0]])
        plt.show()

        # Visual Plots
        max_visual = 5

        time_idx = list(range(0, time_step, int(time_step / max_visual)))
        neur_idx = np.random.randint(mem_rec_hist.shape[1], size=max_visual)

        gs = GridSpec(max_visual, max_visual)
        plt.figure(figsize=(30, 20))

        gs = GridSpec(max_visual, max_visual)
        plt.figure(figsize=(30, 20))

        # Draw Time based mems
        counter = 0
        for n in neur_idx:
            for t in time_idx:
                if counter == 0:
                    a0 = ax = plt.subplot(gs[counter])
                else:
                    ax = plt.subplot(gs[counter], sharey=a0)
                ax.imshow(mem_rec_hist[t, n, :, :], cmap=plt.cm.gray_r, origin="upper", aspect='auto')
                plt.title('t(%d) - n(%d)' % (t, n))
                counter += 1
        plt.show()

        # Draw  Filters
        gs = GridSpec(3, 20)
        plt.figure(figsize=(10, 10))

        counter = 0
        for c_output in range(self.output_channels):
            for c_input in range(self.input_channels):
                if counter == 0:
                    a0 = ax = plt.subplot(gs[counter])
                else:
                    ax = plt.subplot(gs[counter], sharey=a0)
                ax.imshow(self.w.detach().cpu().numpy()[c_output, c_input, :, :], cmap=plt.cm.gray_r, origin="upper", aspect='equal')
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                # plt.title('in(%d) - out(%d)' % (t, n))
                counter += 1

                if counter >= 60:
                    break
            if counter >= 60:
                break
        plt.show()
