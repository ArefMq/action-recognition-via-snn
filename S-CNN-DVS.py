#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Essential Imports
import numpy as np
import torch
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split


# In[2]:


# Local imports
from utils import plot_spk_rec, plot_mem_rec, generate_random_silence_files

from scnn import SNN, SpikingDenseLayer, SpikingConv2DLayer, SpikingConv3DLayer
from scnn.heaviside import SurrogateHeaviside

from scnn.optim import RAdam


# In[3]:


# Tools Import
from data.data_augmentor import data_augment, batchify
from tools.time_expector import TimeExpector
from tools.notify import notify
te = TimeExpector()

def print_progress(msg, value, width=80, a='=', b='>', c='.'):
    print('\r%s [%s%s%s] %d%%' % (msg, a*int(value*width), b, c*int((1.-value)*width), value*100), end='')


# In[4]:


batch_size = 16
nb_epochs = 2


# In[5]:


# Check whether a GPU is available
if torch.cuda.is_available():
    print('using cuda...')
    device = torch.device("cuda")     
else:
    print('using cpu...')
    device = torch.device("cpu")
    
dtype = torch.float


# In[6]:


# FIXME
my_laptop = False
if my_laptop:
    CACHE_FOLDER_PATH = "/Users/aref/dvs-dataset/Cached"
    DATASET_FOLDER_PATH = "/Users/aref/dvs-dataset/DvsGesture"
else:
    CACHE_FOLDER_PATH = "/home/aref/dataset/dvs-dataset"
    DATASET_FOLDER_PATH = "/home/aref/dataset/dvs-dataset"

    
def load_data(trail):
    if trail.startswith('acc'):
        max_augmentation = 1
        augmentation = False
    else:
        max_augmentation = 3 if trail == 'train' else 1
        augmentation = True
    
    trail = trail.replace('acc_', '')
    return batchify(
        trail,
        DATASET_FOLDER_PATH,
        CACHE_FOLDER_PATH,
        condition_limit=['natural'],
        batch_size=batch_size,
        augmentation=augmentation,
        max_augmentation=max_augmentation,
        frame=20
    )

# calculate train dataset size
dataset_size = 0.
for x_batch, y_batch in load_data('train'):
    dataset_size += 1.
    if dataset_size % 64 == 1:
        print('\rpre-processing dataset: %d' % dataset_size, end='')
print('\rpre-processing dataset: %d' % dataset_size)


# In[7]:




# In[8]:


def debug_print(val, name, show_data=False, pytorch=True):
    print('%s' % name, val.shape)
    if show_data:
        print(val)
    if pytorch:
        print('min=%.2f | mean=%.2f | max=%.2f' % (torch.min(val), torch.mean(val), torch.max(val)))
    else:
        print('min=%.2f | mean=%.2f | max=%.2f' % (np.min(val), np.mean(val), np.max(val)))
    print('\n---------------------------------------------------------------\n')


network = SNN().to(device, dtype)


tau_mem = 10e-3
tau_syn = 5e-3
time_step = 1e-3
beta = float(np.exp(-time_step / tau_mem))
weight_scale = 7*(1.0 - beta)


# network.add_layer(NewSpiker,
#     input_shape=4096,
#     output_shape=128,
                  
#     w_init_mean=0.0,
#     w_init_std=weight_scale
# )

network.add_conv3d(input_shape=(64,64),
                   output_shape=(64,64),
                   input_channels=1,
                   output_channels=32,
                   kernel_size=(1,3,3),
                   dilation=(1,1,1),
                   lateral_connections=False,
)

# network.add_layer(SpikingPool2DLayer, kernel_size=(2,2), output_channels=32)
network.add_pool2d(kernel_size=(2,2), output_channels=32)


# network.add_dense(
#     input_shape=4096,
#     output_shape=256,
#    w_init_mean=0.006,
# #     w_init_std=.96,
#     lateral_connections=True
# )

# network.add_layer(SpikingDenseLayer,
#     output_shape=256
# )

# network.add_layer(SpikingDenseLayer,
#     output_shape=128,
#     w_init_mean=.19
# )

network.add_readout(output_shape=12,
                    time_reduction="max" # mean or max
)

network.compile()



lr = 1e-3
weight_decay = 1e-5
reg_loss_coef = 0.1

train_dl = load_data('train')
valid_dl = load_data('test')

# opt = RAdam(network.get_trainable_parameters())
opt = torch.optim.SGD(network.get_trainable_parameters(), lr=lr, momentum=0.9)
network.fit(load_data, optimizer=opt, dataset_size=dataset_size)

train_accuracy = network.compute_classification_accuracy(train_dl)
print("Train accuracy=%.3f"%(train_accuracy))
test_accuracy = network.compute_classification_accuracy(valid_dl)
print("Test accuracy=%.3f"%(test_accuracy))




