import torchvision
import torchvision.transforms as transforms
from spikenet.image_to_spike_convertor import ImageToSpikeConvertor


class SpikingMNISTDataLoader(ImageToSpikeConvertor):
    DEFAULT_FRAME_SIZE = (28, 28)
    CHANNELS = 1

    def __init__(self, *args, **kwargs):
        self.frame_size = kwargs.pop("frame_size", self.DEFAULT_FRAME_SIZE)
        if isinstance(self.frame_size, int):
            self.frame_size = (self.frame_size, self.frame_size)

        _transforms = []
        if self.frame_size != self.DEFAULT_FRAME_SIZE:
            _transforms.append(torchvision.transforms.Resize(self.frame_size))
        _transforms.append(transforms.ToTensor())

        super().__init__(
            *args,
            train_data=torchvision.datasets.MNIST(
                root="./~data/mnist",
                train=True,
                transform=torchvision.transforms.Compose(_transforms),
                download=True,
            ),
            test_data=torchvision.datasets.MNIST(
                root="./~data/mnist",
                train=False,
                transform=torchvision.transforms.Compose(_transforms),
            ),
            **kwargs,
        )

    def x_transform(self, x):
        x = super().x_transform(x)
        res = x.reshape(-1, self.CHANNELS, self.time_scale, *self.frame_size)
        return res
