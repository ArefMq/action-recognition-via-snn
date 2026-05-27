from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class LayerPlottable:
    """Mixin for spiking layers. Adds plotting methods whose implementation
    lives entirely in spikenet.visual.layer_viz."""

    def plot_weights(self, ax: plt.Axes | None = None) -> None:
        from spikenet.visual.layer_viz import LayerVisualizer

        LayerVisualizer(self).plot_weights(ax)

    def plot_activity(self, ax: plt.Axes | None = None) -> None:
        from spikenet.visual.layer_viz import LayerVisualizer

        LayerVisualizer(self).plot_activity(ax)

    def plot_membrane(self, ax: plt.Axes | None = None) -> None:
        from spikenet.visual.layer_viz import LayerVisualizer

        LayerVisualizer(self).plot_membrane(ax)


class NetworkPlottable:
    """Mixin for Network. Adds plotting methods whose implementation
    lives entirely in spikenet.visual.network_viz."""

    def plot_training(
        self,
        history: list[dict[str, float]],
        axes: tuple[plt.Axes, plt.Axes] | None = None,
    ) -> None:
        from spikenet.visual.network_viz import NetworkVisualizer

        NetworkVisualizer(self).plot_training(history, axes)

    def plot_weight_distributions(self, axes: list[plt.Axes] | None = None) -> None:
        from spikenet.visual.network_viz import NetworkVisualizer

        NetworkVisualizer(self).plot_weight_distributions(axes)

    def plot_layer_activity(self, ax: plt.Axes | None = None) -> None:
        from spikenet.visual.network_viz import NetworkVisualizer

        NetworkVisualizer(self).plot_layer_activity(ax)


class DataPlottable:
    """Mixin for DataLoader. Adds plotting methods whose implementation
    lives entirely in spikenet.visual.data_viz."""

    def show_sample(self, trail: str = "train", ax: plt.Axes | None = None) -> None:
        from spikenet.visual.data_viz import DataVisualizer

        DataVisualizer(self).show_sample(trail, ax)

    def plot_spike_raster(self, trail: str = "train", ax: plt.Axes | None = None) -> None:
        from spikenet.visual.data_viz import DataVisualizer

        DataVisualizer(self).plot_spike_raster(trail, ax)

    def plot_histogram(self, trail: str = "train", ax: plt.Axes | None = None) -> None:
        from spikenet.visual.data_viz import DataVisualizer

        DataVisualizer(self).plot_histogram(trail, ax)
