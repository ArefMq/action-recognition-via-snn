from typing import Any

import numpy as np
import torch
from spikenet.layers.neuron_base import NeuronBase
from spikenet.tools.heaviside import SurrogateHeaviside


class SpikingNeuron(NeuronBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.spike_fn = kwargs.get("spike_fn", SurrogateHeaviside.apply)
        self.__time_scale = None
        self.__history: dict[str, np.ndarray] = dict()
        self.__snapshots: dict[str, list[Any]] = dict()

    def _set_time_scale(self, time_scale: int) -> None:
        self.__time_scale = time_scale

    def _history(self, key: str, index: int, value: np.ndarray | torch.Tensor) -> None:
        if key not in self.__history:
            # batch_size, time, reset_of_the_shape
            self.__history[key] = np.zeros((value.shape[0], self.__time_scale, *value.shape[1:]))
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        self.__history[key][:, index, :] = value

    def _append_snapshot(self, key: str, value: Any) -> None:
        if key not in self.__snapshots:
            self.__snapshots[key] = []
        self.__snapshots[key].append(value)

    def snapshot_cycle(self) -> None:
        ...

    def _reset_history(self) -> None:
        self.__history = dict()

    def get_history(self, key: str) -> np.ndarray:
        return self.__history[key]
        
    def get_snapshots(self, key: str) -> list[Any]:
        return self.__snapshots[key]

