# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
#
# from scnn.default_configs import *
#
#
# class SpikingConv3DLayer(torch.nn.Module):
#     IS_CONV = True
#     IS_SPIKING = True
#     HAS_PARAM = True
#
#     def __init__(self, input_shape, input_channels, output_shape=None,
#                  output_channels=1, kernel_size=3, dilation=1,
#                  spike_fn=None, w_init_mean=W_INIT_MEAN, w_init_std=W_INIT_STD, recurrent=False,
#                  lateral_connections=False,
#                  eps=EPSILON, stride=(1, 1, 1), flatten_output=False):
#
#         super(SpikingConv3DLayer, self).__init__()
#
#         self.kernel_size = np.array(kernel_size)
#         self.dilation = np.array(dilation)
#         self.stride = np.array(stride)
#         self.input_channels = input_channels
#         self.input_shape = input_shape
#
#         self.output_channels = output_channels
#         self.output_shape = output_shape if output_shape is not None else input_shape
#         self.spike_fn = spike_fn
#         self.recurrent = recurrent
#         self.lateral_connections = lateral_connections
#         self.eps = eps
#
#         self.flatten_output = flatten_output
#         self.w_init_mean = w_init_mean
#         self.w_init_std = w_init_std
#
#         self.w = torch.nn.Parameter(torch.empty((output_channels, input_channels, *kernel_size)), requires_grad=True)
#         if recurrent:
#             self.v = torch.nn.Parameter(torch.empty((output_channels, output_channels)), requires_grad=True)
#         self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
#         self.b = torch.nn.Parameter(torch.empty(output_channels), requires_grad=True)
#
#         self.reset_parameters()
#         self.clamp()
#
#         self.spk_rec_hist = None
#         self.mem_rec_hist = None
#         self.training = True
#
#     def get_trainable_parameters(self, lr=None, weight_decay=None):
#         res = [
#             {'params': self.w},
#             {'params': self.b},
#             {'params': self.beta},
#         ]
#
#         if self.recurrent:
#             res.append({'params': self.v})
#         if lr is not None:
#             for r in res:
#                 r['lr'] = lr
#         if weight_decay is not None:
#             res[0]['weight_decay'] = weight_decay
#         return res
#
#     def serialize(self):
#         return {
#             'type': 'conv3d',
#             'params': {
#                 'kernel_size': self.kernel_size,
#                 'dilation': self.dilation,
#                 'stride': self.stride,
#                 'output_channels': self.output_channels,
#
#                 'recurrent': self.recurrent,
#                 'lateral_connections': self.lateral_connections,
#
#                 'w_init_mean': self.w_init_mean,
#                 'w_init_std': self.w_init_std,
#             }
#         }
#
#     def serialize_to_text(self):
#         # FIXME: re-write this
#         return 'C3(' + str(self.output_channels) \
#                      + (',k' if self.kernel_size[0] == 1 else ',K') + str(self.kernel_size[1]) \
#                      + (',l' if self.lateral_connections else '') \
#                      + (',r' if self.recurrent else '') \
#                      + ')'
#
#     def forward(self, x):
#         batch_size = x.shape[0]
#         nb_steps = x.shape[2]
#
#         stride = tuple(self.stride)
#         padding = tuple(np.ceil(((self.kernel_size - 1) * self.dilation) / 2).astype(int))
#         conv_x = torch.nn.functional.conv3d(x, self.w, padding=padding,
#                                             dilation=tuple(self.dilation),
#                                             stride=stride)
#         conv_x = conv_x[:, :, :, :self.output_shape[0], :self.output_shape[1]]
#
#         mem = torch.zeros((batch_size, self.output_channels, *self.output_shape), dtype=x.dtype, device=x.device)
#         spk = torch.zeros((batch_size, self.output_channels, *self.output_shape), dtype=x.dtype, device=x.device)
#
#         spk_rec = torch.zeros((batch_size, self.output_channels, nb_steps, *self.output_shape), dtype=x.dtype, device=x.device)
#         mem_rec = torch.zeros((batch_size, self.output_channels, nb_steps, *self.output_shape), dtype=x.dtype)
#
#         if self.lateral_connections:
#             d = torch.einsum("abcde, fbcde -> af", self.w, self.w)
#         b = self.b.unsqueeze(1).unsqueeze(1).repeat((1, *self.output_shape))
#
#         norm = (self.w ** 2).sum((1, 2, 3, 4))
#
#         for t in range(nb_steps):
#             if self.lateral_connections:
#                 rst = torch.einsum("abcd,be ->aecd", spk, d)
#             else:
#                 rst = torch.einsum("abcd,b,b->abcd", spk, self.b, norm)
#
#             input_ = conv_x[:, :, t, :, :]
#             if self.recurrent:
#                 input_ = input_ + torch.einsum("abcd,be->aecd", spk, self.v)
#
#             mem = (mem - rst) * self.beta + input_ * (1. - self.beta)
#             mthr = torch.einsum("abcd,b->abcd", mem, 1. / (norm + self.eps)) - b
#             spk = self.spike_fn(mthr)
#
#             spk_rec[:, :, t, :, :] = spk
#             mem_rec[:, :, t, :, :] = mem.detach().cpu()
#
#         self.spk_rec_hist = spk_rec.detach().cpu().numpy()
#         self.mem_rec_hist = mem_rec.numpy()  # FIXME: do this refactor for other layers as well
#
#         if self.flatten_output:
#             output = torch.transpose(spk_rec, 1, 2).contiguous()
#             output = output.view(batch_size, nb_steps, self.output_channels * np.prod(self.output_shape))
#         else:
#             output = spk_rec
#
#         return output
#
#     def reset_parameters(self):
#         torch.nn.init.normal_(self.w, mean=self.w_init_mean,
#                               std=self.w_init_std * np.sqrt(1. / (self.input_channels * np.prod(self.kernel_size))))
#         if self.recurrent:
#             torch.nn.init.normal_(self.v, mean=self.w_init_mean,
#                                   std=self.w_init_std * np.sqrt(1. / self.output_channels))
#         torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
#         torch.nn.init.normal_(self.b, mean=1., std=0.01)
#
#     def clamp(self):
#         self.beta.data.clamp_(0., 1.)
#         self.b.data.clamp_(min=0.)
#
#     def draw(self, batch_id=0):
#         spk_rec_hist = self.spk_rec_hist[batch_id]
#         mem_rec_hist = self.mem_rec_hist[batch_id]
#
#         time_step = mem_rec_hist.shape[1]
#         channels = mem_rec_hist.shape[0]
#         rest_shape = mem_rec_hist.shape[2:]
#
#         tmp_spk = np.zeros((time_step, channels, *rest_shape))
#         tmp_mem = np.zeros((time_step, channels, *rest_shape))
#         for i in range(time_step):
#             tmp_spk[i, :, :, :] = spk_rec_hist[:, i, :, :]
#             tmp_mem[i, :, :, :] = mem_rec_hist[:, i, :, :]
#         spk_rec_hist = tmp_spk
#         mem_rec_hist = tmp_mem
#
#         flat_spk = np.reshape(spk_rec_hist, (time_step, channels * np.prod(mem_rec_hist.shape[2:])))
#         flat_mem = np.reshape(mem_rec_hist, (time_step, channels * np.prod(mem_rec_hist.shape[2:])))
#
#         # Plot Flats
#         max_flats = 25
#         if flat_mem.shape[1] > max_flats:
#             inx = np.random.randint(flat_mem.shape[1], size=max_flats)
#             flat_spk = flat_spk[:, inx]
#             flat_mem = flat_mem[:, inx]
#
#         for i in range(flat_mem.shape[1]):
#             plt.plot(flat_mem[:, i], label='mem')
#         plt.xlabel('Time')
#         plt.ylabel('Membrace Potential')
#         plt.show()
#
#         plt.plot(flat_spk, '.')
#         plt.xlabel('Time')
#         plt.ylabel('Spikes')
#         plt.show()
#
#         plt.matshow(flat_spk, origin="upper", aspect='auto')
#         plt.xlabel('Neuron')
#         plt.ylabel('Spike Time')
#         plt.axis([-1, flat_spk.shape[1], -1, flat_spk.shape[0]])
#         plt.show()
#
#         # Visual Plots
#         max_visual = 5
#
#         time_idx = list(range(0, time_step, int(time_step / max_visual)))
#         neur_idx = np.random.randint(mem_rec_hist.shape[1], size=max_visual)
#
#         gs = GridSpec(max_visual, max_visual)
#         plt.figure(figsize=(30, 20))
#
#         gs = GridSpec(max_visual, max_visual)
#         plt.figure(figsize=(30, 20))
#
#         # Draw Time based mems
#         counter = 0
#         for n in neur_idx:
#             for t in time_idx:
#                 if counter == 0:
#                     a0 = ax = plt.subplot(gs[counter])
#                 else:
#                     ax = plt.subplot(gs[counter], sharey=a0)
#                 ax.imshow(mem_rec_hist[t, n, :, :], cmap=plt.cm.gray_r, origin="upper", aspect='auto')
#                 plt.title('t(%d) - n(%d)' % (t, n))
#                 counter += 1
#         plt.show()
#
#         # Draw  Filters
#         gs = GridSpec(3, 20)
#         plt.figure(figsize=(10, 10))
#
#         counter = 0
#         for c_output in range(self.output_channels):
#             for c_input in range(self.input_channels):
#                 if counter == 0:
#                     a0 = ax = plt.subplot(gs[counter])
#                 else:
#                     ax = plt.subplot(gs[counter], sharey=a0)
#                 ax.imshow(self.w.detach().cpu().numpy()[c_output, c_input, 0, :, :], cmap=plt.cm.gray_r, origin="upper", aspect='equal')
#                 ax.set_yticklabels([])
#                 ax.set_xticklabels([])
#                 # plt.title('in(%d) - out(%d)' % (t, n))
#                 counter += 1
#
#                 if counter >= 60:
#                     break
#             if counter >= 60:
#                 break
#         plt.show()
