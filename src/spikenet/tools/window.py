import numpy as np

Window3D = tuple[int, int, int]


def window_to_and_array(window: Window3D | int, flat: bool = True) -> np.ndarray:
    """
    Convert a window to an np.ndarray.

    Args:
        window: The window to convert.
        flat: Whether to return a flat array.

    Returns:
        The window as an array.

    Examples:
        >>> window_to_and_array((2, 3, 4))
        array([2, 3, 4])

        >>> window_to_and_array(5)
        array([1, 5, 5])

        >>> window_to_and_array(5, flat=False)
        array([5, 5, 5])
    """
    if isinstance(window, int):
        if flat:
            return np.array([1, window, window])
        return np.array([window, window, window])
    return np.array(window)
