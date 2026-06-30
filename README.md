# SpikeNet

A PyTorch framework for building and training Spiking Neural Networks (SNNs). Uses Leaky Integrate-and-Fire (LIF) neuron dynamics with surrogate gradient backpropagation — so you get biologically plausible spike-based computation with standard gradient descent.

Built as part of my thesis on action recognition via SNNs.
- [Original thesis (Farsi)](docs/thesis_fa.pdf)
- [English translation](docs/thesis_en.pdf)

---

## Install

Requires Python ≥ 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/aref-mehr/action-recognition-via-snn
cd action-recognition-via-snn
uv sync
```

---

## Quick start

### Standard (ReLU) network

```python
from torch.nn import Linear, ReLU
from spikenet.data import DataLoader
from spikenet.network import Network

net = Network(name="Simple Network")
net.add_layer(Linear(784, 500)).add_layer(ReLU()).add_layer(Linear(500, 10))
net.compiled()

history = net.train(data, epochs=20)
metrics = net.test(data)
```

### Spiking network

```python
from spikenet.network import Network
from spikenet.layers import SpikingDenseLayer, SpikingConv2D, SpikingPoolingLayer, Flatten
from spikenet.functions.time_reduction import max_membrane_potential

net = Network(epochs=3, learning_rate=1e-3)
net += SpikingConv2D(16, in_features=1)
net += SpikingPoolingLayer()
net += Flatten()
net += SpikingDenseLayer(128)
net += SpikingDenseLayer(10, time_reduction=max_membrane_potential)

net.compiled(input_features=1, output_features=10)
net.initialize_parameters()

history = net.train(data)
metrics = net.test(data)
```

See [examples/](examples/) for runnable scripts.

---

## > [Notebooks](notebooks/) — start here for a guided walkthrough
