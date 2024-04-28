from typing import Any, Generator
import torch
import torch.utils.data as data


class DataLoader:
    def __init__(
        self,
        train_data: data.Dataset | None = None,
        test_data: data.Dataset | None = None,
        batch_size: int = 128,
        shuffle_train: bool = True,
        shuffle_test: bool = False,
    ):
        # TODO: implement test-train-split if the test_data is None
        self.train_data: data.Dataset = train_data or self.get_train_data()
        self.test_data: data.Dataset = test_data or self.get_test_data()
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.shuffle_test = shuffle_test
        self.__train_loader = self.train_dataloader()
        self.__test_loader = self.test_dataloader()
        self.__shape: tuple[torch.Size, torch.Size] = None  # type: ignore

    def len(self, trail: str = "train") -> int:
        return len(self.train_data) if trail == "train" else len(self.test_data)

    def get_train_data(self) -> data.Dataset:
        raise NotImplementedError

    def get_test_data(self) -> data.Dataset:
        raise NotImplementedError

    # Ready to be overridden
    def x_transform(self, x: Any) -> Any:
        return x

    def y_transform(self, y: Any) -> Any:
        return y

    def train_dataloader(self) -> data.DataLoader:
        self.__train_loader = torch.utils.data.DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
        )
        return self.__train_loader

    def test_dataloader(self) -> data.DataLoader:
        self.__test_loader = torch.utils.data.DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle_test,
        )
        return self.__test_loader

    def __call__(self, trail: str) -> Generator[Any, Any, None]:
        for x_data, y_data in (
            self.__train_loader if trail == "train" else self.__test_loader
        ):
            yield self.x_transform(x_data), self.y_transform(y_data)

    def sample(
        self,
        trail: str = "train",
        random_sample: bool = True,
        apply_transform: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xb, yb = next(self(trail))
        idx = torch.randint(0, self.batch_size, (1,)).item() if random_sample else 0
        if apply_transform:
            return self.x_transform(xb[idx]), self.y_transform(yb[idx])
        return xb[idx], yb[idx]

    @property
    def shape(self) -> tuple[torch.Size, torch.Size]:
        if self.__shape is None:
            x, y = self.sample()
            self.__shape = (x.shape, y.shape)
        return self.__shape

    def describe(self) -> None:
        xshape, yshape = self.shape
        print("shape:")
        print(f"  - x: {xshape}")
        print(f"  - y: {yshape}")
        print(f"  - batch_size: {self.batch_size}")

    def __repr__(self) -> str:
        return f"DataLoader: {self.train_data}, {self.test_data}"
