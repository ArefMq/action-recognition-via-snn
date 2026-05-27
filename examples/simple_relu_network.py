import torchvision
import torchvision.transforms as transforms
from torch.nn import Linear, ReLU

from spikenet.data import DataLoader
from spikenet.network import Network
from spikenet.visual import plot_training_history


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class MyData(DataLoader):
    def __init__(self):
        super().__init__(
            train_data=torchvision.datasets.MNIST(
                root="./~data/mnist",
                train=True,
                transform=transforms.ToTensor(),
                download=True,
            ),
            test_data=torchvision.datasets.MNIST(
                root="./~data/mnist",
                train=False,
                transform=transforms.ToTensor(),
            ),
        )

    def x_transform(self, x):
        # Transform data into 1D array
        return x.reshape(-1, 28 * 28)


data = MyData()


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
net = Network(name="Simple Network")
net.add_layer(Linear(784, 500)).add_layer(ReLU()).add_layer(Linear(500, 10))
net.summarise()


# ---------------------------------------------------------------------------
# Compile & initialise
# ---------------------------------------------------------------------------
net.compiled()


# ---------------------------------------------------------------------------
# Train & evaluate
# ---------------------------------------------------------------------------
history = net.train(data, epochs=20)
metrics = net.test(data)


print(f"\nFinal train loss : {history[-1]['loss']:.4f}")
print(f"Final train acc  : {history[-1]['accuracy']:.4f}")
print(f"Test  loss       : {metrics['loss']:.4f}")
print(f"Test  acc        : {metrics['accuracy']:.4f}")

plot_training_history(history, test_metrics=metrics)
