#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch

from scnn import SNN
from scnn.optim import RAdam

from data.data_augmentor import data_augment, batchify


batch_size = 16
nb_epochs = 6


# Check whether a GPU is available
if torch.cuda.is_available():
    print('using cuda...')
    device = torch.device("cuda")     
else:
    print('using cpu...')
    device = torch.device("cpu")    
dtype = torch.float


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




network = SNN(device=device, dtype=dtype)


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
                   output_channels=128,
                   kernel_size=(1,5,5),
                   dilation=(1,1,1),
                   lateral_connections=False,
)

# network.add_layer(SpikingPool2DLayer, kernel_size=(2,2), output_channels=32)
network.add_pool2d(kernel_size=(4,4), output_channels=128)


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
network = network.to(network.device, network.dtype) # FIXME: this is a bug, fix it!



# opt = RAdam(network.get_trainable_parameters())
opt = torch.optim.SGD(network.get_trainable_parameters(), lr=1e-3, momentum=0.9)
network.fit(load_data, epochs=nb_epochs, optimizer=opt, dataset_size=dataset_size)

print('\n----------------------------------------')
train_accuracy = network.compute_classification_accuracy(load_data('train'))
print("Final Train Accuracy=%.2f%%"%(train_accuracy * 100.))
test_accuracy = network.compute_classification_accuracy(load_data('test'))
print("Final Test Accuracy=%.2f%%"%(test_accuracy * 100.))


