from spikenet.dataset.spiking_mnist import SpikingMNISTDataLoader
from spikenet.functions.time_reduction import spike_rate
from spikenet.layers import Flatten, SpikingDenseLayer
from spikenet.network.network import Network
from spikenet.visual import plot_network_activity

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# Rate-coded MNIST at 14x14. Each image is converted to a spike train of
# shape (1 channel, 16 time steps, 14, 14).
data = SpikingMNISTDataLoader(frame_size=14, time_scale=16, batch_size=64)

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
net = Network(epochs=3, learning_rate=1e-4)

net += Flatten()
net += SpikingDenseLayer(128, b_init_mean=4.0)
net += SpikingDenseLayer(10, time_reduction=spike_rate, w_init_mean=0.001)

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
net.dry_run(data)

# ---------------------------------------------------------------------------
# Train & evaluate
# ---------------------------------------------------------------------------
m = net.train(data, epochs=2, epoch_callback=lambda ep, me: plot_network_activity(net, ep, me))
m += net.train(data, learning_rate=1e-4, epochs=1)

m.print()
m.plot()
