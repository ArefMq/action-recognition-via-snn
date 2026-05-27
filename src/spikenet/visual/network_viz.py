from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from spikenet.network.network import Network


class NetworkVisualizer:
    """Visualisation at the whole-network level: training curves,
    weight distributions, and a layer-by-layer activity summary."""

    def __init__(self, network: Network) -> None:
        self.network = network

    # ------------------------------------------------------------------
    # Training history
    # ------------------------------------------------------------------

    def plot_training(
        self,
        history: list[dict[str, float]],
        axes: tuple[plt.Axes, plt.Axes] | None = None,
    ) -> None:
        """Loss and accuracy curves from the list returned by Network.train()."""
        if not history:
            raise ValueError("History is empty — call net.train(data) first.")

        epochs = [m["epoch"] for m in history]
        losses = [m["loss"] for m in history]
        accuracies = [m["accuracy"] for m in history]

        created = axes is None
        if created:
            fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
        else:
            ax_loss, ax_acc = axes

        ax_loss.plot(epochs, losses, "o-", color="steelblue", markersize=4, linewidth=1.5)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Training loss")
        ax_loss.grid(True, alpha=0.3)

        ax_acc.plot(epochs, accuracies, "o-", color="seagreen", markersize=4, linewidth=1.5)
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1)
        ax_acc.set_title("Training accuracy")
        ax_acc.grid(True, alpha=0.3)

        if created:
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    # Weight distributions
    # ------------------------------------------------------------------

    def plot_weight_distributions(self, axes: list[plt.Axes] | None = None) -> None:
        """One histogram per layer that has initialised weights."""
        from spikenet.layers.neuron_base import NeuronBase

        layers_with_weights = [
            layer
            for layer in self.network._layers
            if isinstance(layer, NeuronBase) and getattr(layer, "w", None) is not None
        ]

        if not layers_with_weights:
            raise RuntimeError("No initialised weights found. Call initialize_parameters() first.")

        n = len(layers_with_weights)
        created = axes is None
        if created:
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
            if n == 1:
                axes = [axes]

        for ax, layer in zip(axes, layers_with_weights, strict=False):
            w = layer.w.detach().cpu().flatten().numpy()
            ax.hist(w, bins=60, color="steelblue", alpha=0.8, edgecolor="none")
            ax.axvline(w.mean(), color="tomato", linewidth=1.5, label=f"mean={w.mean():.3f}")
            ax.set_title(layer.name)
            ax.set_xlabel("Weight value")
            ax.legend(fontsize=8)

        if created:
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    # Layer-by-layer activity summary
    # ------------------------------------------------------------------

    def plot_layer_activity(self, ax: plt.Axes | None = None) -> None:
        """Bar chart of mean spike rate per layer after the last forward pass."""
        from spikenet.layers.spiking_base import SpikingNeuron

        active_layers = [
            layer for layer in self.network._layers if isinstance(layer, SpikingNeuron) and layer._spike_rec is not None
        ]

        if not active_layers:
            raise RuntimeError("No spike records found. Run a forward pass first.")

        names = [layer.name for layer in active_layers]
        rates = [layer._spike_rec.float().mean().item() for layer in active_layers]

        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(max(4, len(active_layers) * 1.4), 3.5))

        bars = ax.bar(names, rates, color="steelblue", width=0.6)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.set_ylabel("Mean spike rate")
        ax.set_title("Layer activity summary")
        ax.set_ylim(0, max(rates) * 1.25 if rates else 1)
        plt.xticks(rotation=25, ha="right")

        if created:
            plt.tight_layout()
            plt.show()
