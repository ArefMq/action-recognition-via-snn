import torchvision
import torchvision.transforms as transforms
from spikenet.image_to_spike_convertor import ImageToSpikeConvertor


class SpikingMNISTDataLoader(ImageToSpikeConvertor):
    RESIZE_SIZE = (16, 16)

    def __init__(self, *args, **kwargs):
        if "resize_to" in kwargs:
            self.RESIZE_SIZE = kwargs.pop("resize_to")

        super().__init__(
            *args,
            train_data=torchvision.datasets.MNIST(
                root="./data",
                train=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(self.RESIZE_SIZE),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            ),
            test_data=torchvision.datasets.MNIST(
                root="./data",
                train=False,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(self.RESIZE_SIZE),
                        transforms.ToTensor(),
                    ]
                ),
            ),
            **kwargs,
        )

    def x_transform(self, x):
        x = super().x_transform(x)
        res = x.reshape(-1, self.time_scale, self.RESIZE_SIZE[0] * self.RESIZE_SIZE[1])
        return res


spiking_mnist = SpikingMNISTDataLoader()