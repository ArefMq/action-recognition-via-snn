import torch
import torch.utils.data as data
from torch import Tensor

from .dataloader import DataLoader


class ImageToSpikeConverter(DataLoader):
    """
    This class is a base for data loaders that convert images to spiking trains.
    The main idea is apply either rate or time based coding to the input images,
    and add an extra dimension for the time steps. The Spikes will be represented
    as 1 and no-spikes as 0.

    Args:
        train_data: the training data. Default is None
        test_data: the testing data. Default is None
        batch_size: the batch size. Default is 128
        shuffle_train: Shuffle the training data. Default is True
        shuffle_test: Shuffle the testing data. Default is False
        time_scale: the number of time steps for the spiking neurons. Default is 16
        background_gain: The minimum value for the background. Default is 0.02
        soft_rate_limit: The maximum value for the soft rate. Default is 0.8
        apply_softening: whether to apply softening or not. Default is True
        coding_type: the type of coding to be applied. (values: {"rate" | "time"} Default is "rate")
    """

    def __init__(
        self,
        train_data: data.Dataset | None = None,
        test_data: data.Dataset | None = None,
        batch_size: int = 128,
        shuffle_train: bool = True,
        shuffle_test: bool = False,
        time_scale: int = 16,
        background_gain: float = 0.02,
        soft_rate_limit: float = 0.8,
        apply_softening: bool = True,
        coding_type: str = "rate",
    ) -> None:
        super().__init__(
            train_data=train_data,
            test_data=test_data,
            batch_size=batch_size,
            shuffle_train=shuffle_train,
            shuffle_test=shuffle_test,
        )
        self.__time_scale = time_scale
        self.__background_gain = background_gain
        self.__soft_rate_limit = soft_rate_limit
        self.__apply_softening = apply_softening
        self.__coding_type = coding_type

    @property
    def time_scale(self) -> int:
        return self.__time_scale

    def convert_to_time_based_spiking_trains(self, batch_imgs: Tensor) -> Tensor:
        batch_size = batch_imgs.shape[0]
        flat_images = batch_imgs.reshape(batch_size, -1)
        intensities = 1 - flat_images
        t = torch.clamp(
            (intensities * self.__time_scale).to(torch.int),
            min=0,
            max=self.__time_scale,
        )
        spikes = torch.zeros((batch_size, self.__time_scale + 1, flat_images.shape[1]))
        idx = torch.arange(flat_images.shape[1])
        spikes[torch.arange(batch_size)[:, None], t, idx] = 1
        return spikes[:, :-1].reshape(batch_size, self.__time_scale, *batch_imgs.shape)

    def convert_to_rate_based_spiking_trains(self, batch_imgs: Tensor) -> Tensor:
        batch_size = batch_imgs.shape[0]
        flat_images = batch_imgs.reshape(batch_size, -1)
        assert torch.min(flat_images) >= 0 and torch.max(flat_images) <= 1
        rand_tensor = torch.rand(batch_size, self.__time_scale, flat_images.shape[1])
        flat_images = flat_images.unsqueeze(1)
        spikes = (rand_tensor < flat_images).to(torch.float)
        return spikes.reshape(batch_size, self.__time_scale, *batch_imgs.shape[-2:])

    def soften_image(self, img: Tensor) -> Tensor:
        assert 0 <= self.__background_gain <= 1 and 0 <= self.__soft_rate_limit <= 1
        img = img * self.__soft_rate_limit
        return img * (1 - self.__background_gain) + self.__background_gain

    def x_transform(self, x: Tensor) -> Tensor:
        if self.__apply_softening:
            x = self.soften_image(x)
        if self.__coding_type == "time":
            x = self.convert_to_time_based_spiking_trains(x)
        elif self.__coding_type == "rate":
            x = self.convert_to_rate_based_spiking_trains(x)
        else:
            raise ValueError(f"Invalid coding type: {self.__coding_type}")
        return x
