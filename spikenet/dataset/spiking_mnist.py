import torch
import torchvision
import torchvision.transforms as transforms
from spikenet.image_to_spike_convertor import ImageToSpikeConvertor


class SpikingMNISTDataLoader(ImageToSpikeConvertor):
    """
    This class is loading the MNIST dataset from torchvision.datasets.MNIST as a DataLoader which would
    be compatible with the rest of the SpikeNet library. This class, will convert the input images into
    spiking trains based on the given time_scale and coding_type. The default image size is 28x28. If a
    different size is given, the images will be resized.
    """
    DEFAULT_FRAME_SIZE = (28, 28)
    CHANNELS = 1

    def __init__(self, *args, **kwargs) -> None:
        """

        Args:
            frame_size: the size of the input images
            time_scale: the number of time steps for the spiking neurons
            background_gain: the background gain for the softening
            soft_rate_limit: the soft rate limit for the softening
            apply_softening: whether to apply softening or not
            coding_type: the type of coding to be applied
        """
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

    def x_transform(self, x: torch.Tensor) -> torch.Tensor:
        x = super().x_transform(x)
        res = x.reshape(-1, self.CHANNELS, self.time_scale, *self.frame_size)
        return res
