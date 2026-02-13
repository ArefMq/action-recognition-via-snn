import numpy as np

Window3D = tuple[int, int, int]


def window_to_and_array(window: Window3D | int, flat: bool = False) -> np.ndarray:
    if isinstance(window, int):
        if flat:
            return np.array([1, window, window])
        return np.array([window, window, window])
    return np.array(window)
