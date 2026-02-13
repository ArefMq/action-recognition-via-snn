import torch
import torchvision
import torchvision.transforms as transforms
from spikenet.dataloader import DataLoader


class MNISTDataLoader(DataLoader):
    """
    This class is loading the MNIST dataset from torchvision.datasets.MNIST as a DataLoader which would
    be compatible with the rest of the SpikeNet library.
    """
    def __init__(self) -> None:
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

    def x_transform(self, x: torch.Tensor) -> torch.Tensor:
        # This will be called within the DataLoader class, in order to
        # flatten the input data into a 1D tensor.
        return x.reshape(-1, 28 * 28)


mnist_data = MNISTDataLoader()