from typing import Any, Callable, Union

import torch

#FIXME: Refactor this code entirely


class CallbackInterface:
    def __init__(self, next_callback: Union["CallbackInterface", None] = None) -> None:
        self.__next_callback = next_callback

    def __call__(self, call_type: str, **kwargs) -> Any:
        call_type = call_type.replace(".", "_")
        if hasattr(self, call_type):
            getattr(self, call_type)(**kwargs)
        if self.__next_callback is not None:
            self.__next_callback(call_type, **kwargs)

    def train_before(self, **kwargs) -> None: ...

    def train_after(self, **kwargs) -> None: ...

    def train_epoch_before(self, **kwargs) -> None: ...

    def train_epoch_after(self, **kwargs) -> None: ...

    def train_batch_before(self, **kwargs) -> None: ...

    def train_batch_after(self, **kwargs) -> None: ...

    def test_before(self, **kwargs) -> None: ...

    def test_after(self, **kwargs) -> None: ...

    def test_batch_before(self, **kwargs) -> None: ...

    def test_batch_after(self, **kwargs) -> None: ...


CallbackTypes = (
    None | str | CallbackInterface | Callable | list[Callable | CallbackInterface | str]
)


class CallbackFactory:
    @staticmethod
    def parse_callbacks(
        callbacks: CallbackTypes,
    ) -> CallbackInterface:
        if callbacks == "default":
            return DefaultCallback()
        return CallbackInterface()

    @staticmethod
    def default() -> CallbackInterface:
        return DefaultCallback()

    @staticmethod
    def empty() -> CallbackInterface:
        return CallbackInterface()


#TODO: This is progress printer or something
class DefaultCallback(CallbackInterface):
    def __init__(self) -> None:
        super().__init__()
        self.loss = []
        self.epoch_loss = []

    def train_before(self, **kwargs) -> None:
        print("===== Training Started ======")

    def train_after(self, **kwargs) -> None:
        import matplotlib.pyplot as plt

        print("===== Training Finished =====")
        plt.plot(self.loss)
        plt.yscale("log")
        plt.title("Loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.show()

    def test_before(self, **kwargs) -> None:
        print("====== Testing Started ======")

    def test_after(self, **kwargs) -> None:
        total_test_points = kwargs["total_test_points"]
        correct_test_points = kwargs["correct_test_points"]
        acc = correct_test_points / total_test_points * 100
        print(f"Accuracy: {correct_test_points}/{total_test_points} = {acc:.2f}%")
        print("===== Testing Finished ======")

    def train_epoch_before(self, **kwargs) -> None:
        print(f"Epoch {kwargs['epoch']} | ", end="")

    def train_epoch_after(self, **kwargs) -> None:
        epoch_loss_mean = torch.tensor(self.epoch_loss).mean().item()
        print(f" | Loss: {epoch_loss_mean:.4f}")
        self.epoch_loss = []

    def train_batch_before(self, **kwargs) -> None:
        self.epoch_loss = []
        batch_id = kwargs["batch_id"]
        if batch_id % 10 == 0:
            print(f".", end="")

    def train_batch_after(self, **kwargs) -> None:
        loss = kwargs["loss"]
        self.epoch_loss.append(loss)
        self.loss.append(loss)


# TODO: Add network saver callback
# TODO: Add early stopping callback
# TODO: Add tensorboard callback
# TODO: Add notification callback
# TODO: Add progress bar callback