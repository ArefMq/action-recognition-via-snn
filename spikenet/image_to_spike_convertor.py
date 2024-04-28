from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
from spikenet.dataloader import DataLoader


class ImageToSpikeConvertor(DataLoader):
    def __init__(self, *args, **kwargs) -> None:
        self.__time_scale = int(kwargs.pop("time_scale", 16))
        self.__background_gain = float(kwargs.pop("background_gain", 0.02))
        self.__soft_rate_limit = float(kwargs.pop("soft_rate_limit", 0.8))
        self.__apply_softening = bool(kwargs.pop("apply_softening", True))
        self.__coding_type = kwargs.pop("coding_type", "rate")
        super().__init__(*args, **kwargs)

    @property
    def time_scale(self) -> int:
        return self.__time_scale

    def convert_to_time_based_spiking_trains(
        self, batch_imgs: torch.Tensor
    ) -> torch.Tensor:
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

    def convert_to_rate_based_spiking_trains(
        self, batch_imgs: torch.Tensor
    ) -> torch.Tensor:
        batch_size = batch_imgs.shape[0]
        flat_images = batch_imgs.reshape(batch_size, -1)
        assert torch.min(flat_images) >= 0 and torch.max(flat_images) <= 1
        rand_tensor = torch.rand(batch_size, self.__time_scale, flat_images.shape[1])
        flat_images = flat_images.unsqueeze(1)
        spikes = (rand_tensor < flat_images).to(torch.float)
        return spikes.reshape(
            batch_size, self.__time_scale, *batch_imgs.shape[-2:]
        )

    def soften_image(self, img: torch.Tensor) -> torch.Tensor:
        assert 0 <= self.__background_gain <= 1 and 0 <= self.__soft_rate_limit <= 1
        img = img * self.__soft_rate_limit
        img = img * (1 - self.__background_gain) + self.__background_gain
        return img

    def x_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.__apply_softening:
            x = self.soften_image(x)
        if self.__coding_type == "time":
            x = self.convert_to_time_based_spiking_trains(x)
        elif self.__coding_type == "rate":
            x = self.convert_to_rate_based_spiking_trains(x)
        else:
            raise ValueError(f"Invalid coding type: {self.__coding_type}")
        return x


class SpikePlotter:
    @staticmethod
    def plot_sample(dataloader: ImageToSpikeConvertor) -> None:
        import matplotlib.pyplot as plt

        x, y = dataloader.sample()
        x = x.reshape(-1, 28, 28).numpy()
        plt.imshow(x.sum(0))
        plt.title(f"Label: {y.item()}")
        plt.show()

    def animate(
        self,
        data: torch.Tensor,
        vmin: float = 0,
        vmax: float = 1,
        cmap: str = "gray",
        title: str | None = None,
    ):
        fig, ax = plt.subplots()
        if isinstance(data, torch.Tensor):
            data = data.detach().numpy()
        t, w, h = data.shape
        img = ax.imshow(np.ones((w, h)), cmap=cmap, vmin=vmin, vmax=vmax)
        if title:
            ax.set_title(title)

        def init():
            return (img,)

        def update(frame):
            img.set_data(data[frame])
            return (img,)

        ani = FuncAnimation(
            fig, update, frames=np.arange(0, t), init_func=init, blit=False
        )
        res = ani.to_jshtml()
        plt.close()
        return res

    def plot_spikes_history_1d(self, spikes, title: str | None = None):
        plt.figure()
        plt.plot(spikes)
        plt.title(title or "1D Spikes")
        plt.show()
