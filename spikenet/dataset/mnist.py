import torchvision
import torchvision.transforms as transforms
from spikenet.dataloader import DataLoader


class MNISTDataLoader(DataLoader):
    def __init__(self):
        super().__init__(
            train_data=torchvision.datasets.MNIST(
                root="./data",
                train=True,
                transform=transforms.ToTensor(),
                download=True,
            ),
            test_data=torchvision.datasets.MNIST(
                root="./data", train=False, transform=transforms.ToTensor()
            ),
        )

    def x_transform(self, x):
        return x.reshape(-1, 28 * 28)


mnist_data = MNISTDataLoader()