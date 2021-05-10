#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
from random import random
np.seterr(all='raise')


# In[2]:


# setup matplotlib
import matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    SAVE_PLOTS = False
else:
    matplotlib.use('Agg')
    SAVE_PLOTS = True
import matplotlib.pyplot as plt


# In[3]:


from utils import plot_spikes_in_time, print_and_plot_accuracy_metrics, plot_metrics, train_test_split
from scnn import SNN
from scnn.Spike.readin import ReadInLayer
from scnn.optim import RAdam

# from data.data_augmentor import data_augment, batchify, GESTURE_MAPPING
from tools.time_expector import TimeExpector
from tools.notify import notify
time_expector = TimeExpector()


# In[4]:


#======================================== Data ========================================
# from navgesture import load_all, classes, num_of_classes
# DATA_PATH = './drive/My Drive/Colab Notebooks/dataset/navgesture/user01/*'
# DATA_PATH = './navgesture/user*/*'

from data import load_all, classes, num_of_classes
DATA_PATH = './dvs/'
# DATA_PATH = '/media/aref/TeraDisk/Workspace/dvs'

#===================================== Parameters =====================================
BATCH_SIZE = 1
IMAGE_SIZE = (128, 128)
IMAGE_SCALE = (.4, .4)
FRAMES = 20
FRAME_LENGTH = 5000 # in timestamps
POLARITY_MODE = 'accumulative' # 'twolayer' 'onelayer' 'ignore'

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 30
NET_STR = 'c(64,k7) -> c(128,k7) -> c(256,k7) -> d(256,l) -> d(128,l)'

#====================================== Configs  ======================================
read_in_channels = 2 if POLARITY_MODE in ['accumulative', 'twolayer'] else 1
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
dtype = torch.float
print("Device:", device)


# In[5]:


data_loading_config = {
    'batch_size': BATCH_SIZE,
    'image_size': IMAGE_SIZE,
    'image_scale': IMAGE_SCALE,
    'frames': FRAMES,
    'frame_len': FRAME_LENGTH,
    'polarity_mode': POLARITY_MODE,
    'data_path': DATA_PATH,
#     'max_read_file': 10,
}

x_cache_train = x_cache_test = y_cache_train = y_cache_test = None

def load_cache():
    global x_cache_train, x_cache_test, y_cache_train, y_cache_test
    
    x_cache_data = []
    y_cache_data = []

    for xc, yc in load_all(**data_loading_config):
        x_cache_data.append(xc)
        y_cache_data.append(yc)

    x_cache_train, x_cache_test, y_cache_train, y_cache_test = train_test_split(
        x_cache_data,
        y_cache_data,
        test_size=0.3
    )

    print('[Done]')
    print('Train Dataset Size: %d * %d' % (len(x_cache_train), BATCH_SIZE))
    print('Test Dataset Size: %d * %d' % (len(x_cache_test), BATCH_SIZE))
    print('-------------------------\n')
    if len(x_cache_test) == 0 or len(x_cache_train) == 0:
        raise Exception('not enough data collected')


# In[6]:


def batchify(data, batch_size):
    x_batch = []
    y_batch = []
    for x_chunk, y_chunk in data:
        if len(x_batch) == batch_size:
            yield x_batch, y_batch
            x_batch = []
            y_batch = []
        x_batch.append(x_chunk.to_dense().numpy() if x_chunk.layout == torch.sparse_coo else x_chunk.numpy())
        y_batch.append(y_chunk)

    
def load_data(trail=''):
    if x_cache_train is None or x_cache_test is None:
        load_cache()
    
    trail = trail.replace('acc_', '')
    zipper = zip(x_cache_train, y_cache_train) if trail == 'train' else zip(x_cache_test, y_cache_test)
    for x_data, y_data in batchify(zipper, BATCH_SIZE):
#         yield np.array(x_data)[:, 0, :, :, :, :], np.array(y_data)[:,0]
        yield np.array(x_data).reshape(BATCH_SIZE, FRAMES, read_in_channels, 128, 128), np.array(y_data)


# In[7]:


# #====================================== Network Begining =====================================
# network = SNN(
#     device=device,
#     dtype=dtype,
#     input_layer=ReadInLayer(
#         input_shape=(128,128),
#         input_channels=read_in_channels
#     )
# )

# network.network.time_expector = time_expector
# # network.notifier = notify # FIXME
# # network.add_pool2d(input_shape=(128,128), kernel_size=(2,2), reduction='max')


# #===================================== Network Structure =====================================

# network.parse_str(NET_STR)

# # network.add_conv3d(
# #     output_channels=5,
# #     kernel_size=(1,3,3),
# #     dilation=(1,1,1),
# #     lateral_connections=True,
# #     recurrent=False,
# #     w_init_mean=0.00,
# #     w_init_std=0.05
# # )
# # network.add_pool2d(kernel_size=(2,2), reduction='max')


# # network.add_dense(
# #     output_shape=5,
# #     w_init_mean=0.0,
# #     w_init_std=0.3,
# #     lateral_connections=True,
# #     recurrent=False,
# # #     dropout_prob=0.3,
# # )


# #=================================== Network Finalization ====================================
# network.add_readout(
#     output_shape=num_of_classes,
#     time_reduction="max",

#     w_init_mean=0.0,
#     w_init_std=0.3
# )
# network.compile()
# network.print_summery()
# # print('Network Summery:', network)
# network.plot_one_batch(load_data('train'))


# In[8]:


# result_file = open('./logs/results.log', 'a+') if SAVE_PLOTS else None
# opt = RAdam(network.get_trainable_parameters(LEARNING_RATE, WEIGHT_DECAY))
# # opt = torch.optim.SGD(network.get_trainable_parameters(LEARNING_RATE, WEIGHT_DECAY), lr=LEARNING_RATE, momentum=0.9)

# res_metrics = network.fit(
#     load_data,
#     epochs=EPOCHS,
#     optimizer=opt,
#     result_file=result_file,
#     save_checkpoints=False, #SAVE_PLOTS
# )
# plot_metrics(res_metrics, save_plot_path='./logs/metrics_' if SAVE_PLOTS else None)

# if SAVE_PLOTS:
#     result_file.close()


# In[9]:


# network.save('./logs/save_network.net')
# network.load('./logs/save_network.net')


# In[10]:


# network.plot_one_batch(load_data('test'))
# print_and_plot_accuracy_metrics(
#     network, 
#     load_data('acc_train'), 
#     load_data('acc_test'), 
#     save_plot_path='./logs/truth_' if SAVE_PLOTS else None
# )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional, surrogate, layer
import sys


# In[12]:


class Net(nn.Module):
    def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.T = T

        self.static_conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        self.conv = nn.Sequential(
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),  # 32 * 32

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2)  # 16 * 16

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.7),
            nn.Linear(128 * 16 * 16, 128 * 3 * 3, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            layer.Dropout(0.7),
            nn.Linear(128 * 3 * 3, 128, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.Linear(128, num_of_classes, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )


    def forward(self, x):
        x = self.static_conv(x)

        out_spikes_counter = self.fc(self.conv(x))
        for t in range(1, self.T):
            out_spikes_counter += self.fc(self.conv(x))

        return out_spikes_counter / self.T


# In[13]:


T = FRAMES
tau = 2.
train_epoch = EPOCHS


# In[ ]:






net = Net(tau=tau, T=T).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
train_times = 0
max_test_accuracy = 0

for epoch in range(train_epoch):
    net.train()
    for img, label in load_data('train'):
        print('\rT:  %d' % train_times, end='')
        img = torch.from_numpy(img)[0, :, 1, :, :].view((20,1,128,128))
        img = img.to(device)
        label = torch.from_numpy(label).to(device)
        label_one_hot = F.one_hot(label, num_of_classes).float()

        optimizer.zero_grad()
        out_spikes_counter_frequency = net(img)

        max_over_time = torch.max(out_spikes_counter_frequency, axis=0).values.view([1, num_of_classes])
        loss = F.mse_loss(max_over_time, label_one_hot)
        loss.backward()
        optimizer.step()

        functional.reset_net(net)
        accuracy = (max_over_time.max(1)[1] == label).float().mean().item()
        train_times += 1

    net.eval()
    with torch.no_grad():
        test_sum = 0
        correct_sum = 0
        for img, label in load_data('test'):
            img = torch.from_numpy(img)[0, :, 1, :, :].view((20,1,128,128))
            img = img.to(device)
            label = torch.from_numpy(label).to(device)
            out_spikes_counter_frequency = net(img)

            max_over_time = torch.max(out_spikes_counter_frequency, axis=0).values.view([1, num_of_classes])
            correct_sum += (max_over_time.max(1)[1] == label).float().sum().item()
            test_sum += label.numel()
            functional.reset_net(net)


        test_accuracy = correct_sum / test_sum
        print('Acc: \t %.2f' % test_accuracy, end='')
        if max_test_accuracy < test_accuracy:
            max_test_accuracy = test_accuracy
            print('\t[Max]')
        else:
            print('')

# In[ ]:




