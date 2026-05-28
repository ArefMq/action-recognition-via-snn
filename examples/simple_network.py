from spikenet.dataset.spiking_mnist import SpikingMNISTDataLoader
from spikenet.functions.time_reduction import max_membrane_potential
from spikenet.layers import Flatten, SpikingConv2D, SpikingDenseLayer, SpikingPoolingLayer
from spikenet.network.network import Network
from spikenet.visual import plot_network_activity, plot_training_history

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# Rate-coded MNIST at 14x14. Each image is converted to a spike train of
# shape (1 channel, 16 time steps, 14, 14).
data = SpikingMNISTDataLoader(frame_size=14, time_scale=16, batch_size=64)

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
net = Network(epochs=3, learning_rate=1e-3)

# First conv block: 1 input channel (binary spikes) → 16 feature maps
net += SpikingConv2D(16, in_features=1)
net += SpikingPoolingLayer()

# Bridge: reshape (batch, channels, time, h, w) → (batch, time, features)
net += Flatten()

# Dense hidden layer
net += SpikingDenseLayer(128)

# Output layer: 10 classes. max_membrane_potential collapses the time
# dimension into a single class score per neuron.
net += SpikingDenseLayer(10, time_reduction=max_membrane_potential)

# ---------------------------------------------------------------------------
# Compile & initialise
# ---------------------------------------------------------------------------
# Peek at one batch to learn the full input shape (channels, time, h, w).
# compiled(data_shape=...) runs a dummy forward pass so that Flatten and any
# lazy-init layers resolve their sizes before summarise() is called.
sample_x, _ = next(iter(data("train")))
net.compiled(input_features=1, output_features=10, data_shape=sample_x.shape)
net.initialize_parameters()

net.summarise()

# ---------------------------------------------------------------------------
# Train & evaluate
# ---------------------------------------------------------------------------
history = net.train(data, epochs=10, epoch_callback=lambda ep, m: plot_network_activity(net, ep, m))
metrics = net.test(data)

print(f"\nFinal train loss : {history[-1]['loss']:.4f}")
print(f"Final train acc  : {history[-1]['accuracy']:.4f}")
print(f"Test  loss       : {metrics['loss']:.4f}")
print(f"Test  acc        : {metrics['accuracy']:.4f}")

plot_training_history(history, test_metrics=metrics)
