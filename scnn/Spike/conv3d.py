import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scnn.Spike.conv2d import SpikingConv2DLayer
from scnn.default_configs import *


class SpikingConv3DLayer(SpikingConv2DLayer):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'Conv3D'
        if 'output_shape' not in kwargs:
            kwargs['output_shape'] = kwargs['input_shape']

        if 'stride' not in kwargs:
            kwargs['stride'] = (1, 1, 1)

        if 'kernel_size' not in kwargs:
            kwargs['kernel_size'] = (1, 3, 3)

        if 'dilation' not in kwargs:
            kwargs['dilation'] = (1, 1, 1)

        super(SpikingConv3DLayer, self).__init__(*args, **kwargs)

    def __str__(self):
        # FIXME: re-write this
        return 'C3(' + str(self.output_channels) \
                     + (',k' if self.kernel_size[0] == 1 else ',K') + str(self.kernel_size[1]) \
                     + (',l' if self.lateral_connections else '') \
                     + (',r' if self.recurrent else '') \
                     + ')'

    def forward_function(self, x):
        batch_size = x.shape[0]
        nb_steps = x.shape[2]

        stride = tuple(self.stride)
        padding = tuple(np.ceil(((self.kernel_size - 1) * self.dilation) / 2).astype(int))
        conv_x = torch.nn.functional.conv3d(x, self.w, padding=padding,
                                            dilation=tuple(self.dilation),
                                            stride=stride)
        conv_x = conv_x[:, :, :, :self.output_shape[0], :self.output_shape[1]]

        mem = torch.zeros((batch_size, self.output_channels, *self.output_shape), dtype=x.dtype, device=x.device)
        spk = torch.zeros((batch_size, self.output_channels, *self.output_shape), dtype=x.dtype, device=x.device)

        spk_rec = torch.zeros((batch_size, self.output_channels, nb_steps, *self.output_shape), dtype=x.dtype, device=x.device)
        # mem_rec = torch.zeros((batch_size, self.output_channels, nb_steps, *self.output_shape), dtype=x.dtype)

        if self.lateral_connections:
            d = torch.einsum("abcde, fbcde -> af", self.w, self.w)
        b = self.b.unsqueeze(1).unsqueeze(1).repeat((1, *self.output_shape))

        norm = (self.w ** 2).sum((1, 2, 3, 4))

        for t in range(nb_steps):
            if self.lateral_connections:
                rst = torch.einsum("abcd,be ->aecd", spk, d)
            else:
                rst = torch.einsum("abcd,b,b->abcd", spk, self.b, norm)

            input_ = conv_x[:, :, t, :, :]
            if self.recurrent:
                input_ = input_ + torch.einsum("abcd,be->aecd", spk, self.v)

            mem = (mem - rst) * self.beta + input_ * (1. - self.beta)
            mthr = torch.einsum("abcd,b->abcd", mem, 1. / (norm + EPSILON)) - b
            spk = self.spike_fn(mthr)

            spk_rec[:, :, t, :, :] = spk
            # mem_rec[:, :, t, :, :] = mem.detach().cpu()

        # self.spk_rec_hist = spk_rec.detach().cpu().numpy()
        # self.mem_rec_hist = mem_rec.numpy()  # FIXME: do this refactor for other layers as well

        if self.flatten_output:
            output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = output.view(batch_size, nb_steps, self.output_channels * np.prod(self.output_shape))
        else:
            output = spk_rec

        return output

    def draw(self, batch_id=0, layer_id=None):
        if self.mem_rec_hist is None:
            return

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
        if layer_id is not None:
            plt.title('layer: %s' % layer_id)
        plt.show()

        plt.plot(flat_spk, '.')
        plt.xlabel('Time')
        plt.ylabel('Spikes')
        if layer_id is not None:
            plt.title('layer: %s' % layer_id)
        plt.show()

        plt.matshow(flat_spk, origin="upper", aspect='auto')
        plt.xlabel('Neuron')
        plt.ylabel('Spike Time')
        plt.axis([-1, flat_spk.shape[1], -1, flat_spk.shape[0]])
        if layer_id is not None:
            plt.title('layer: %s' % layer_id)
        plt.show()

        # Visual Plots
        max_visual = 5

        time_idx = list(range(0, time_step, int(time_step / max_visual)))[:max_visual]
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
                plt.title('t(%d) - n(%d) - layer: %s' % (t, n, layer_id if layer_id is not None else ''))
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
                ax.imshow(self.w.detach().cpu().numpy()[c_output, c_input, 0, :, :], cmap=plt.cm.gray_r, origin="upper", aspect='equal')
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                # plt.title('in(%d) - out(%d)' % (t, n))
                counter += 1

                if counter >= 60:
                    break
            if counter >= 60:
                break
        if layer_id is not None:
            plt.title('layer: %s' % layer_id)
        plt.show()
