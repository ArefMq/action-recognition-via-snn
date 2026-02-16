from spikenet.layers import SpikingConv2D, SpikingDenseLayer, SpikingPoolingLayer
from spikenet.network.network import Network

net = Network()

# Input layer with 16 features, kernel=3;
net += SpikingConv2D(16)
net += SpikingPoolingLayer()

# Next layer with 32 features.
net += SpikingConv2D(32)
net += SpikingPoolingLayer()

# A dense layer with 128 features
net += SpikingDenseLayer(128)

# Output layer. Features are dynamically assigned based on
# the number of classes in the dataset.
net += SpikingDenseLayer()
net.summarise()

net.fit()
