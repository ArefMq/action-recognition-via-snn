from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from spikenet.data.dataloader import DataLoader


class DataVisualizer:
    """Visualisation at the data level: samples, spike rasters, and statistics."""

    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader

    # ------------------------------------------------------------------
    # Sample view
    # ------------------------------------------------------------------

    def show_sample(self, trail: str = "train", ax: plt.Axes | None = None) -> None:
        """Display one sample. For spike-encoded data the spike counts are
        accumulated over time to produce a 2-D heatmap."""
        x, y = self.dataloader.sample(trail=trail)
        label = y.item() if hasattr(y, "item") else y

        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(4, 4))

        img = self._to_image(x)
        cmap = "viridis" if x.dim() == 4 else "gray"
        ax.imshow(img, cmap=cmap)
        ax.set_title(f"Label: {label}")
        ax.axis("off")

        if created:
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    # Spike raster
    # ------------------------------------------------------------------

    def plot_spike_raster(self, trail: str = "train", ax: plt.Axes | None = None) -> None:
        """Time-vs-feature raster of one spike-encoded sample.

        Expects the sample to have a time dimension, e.g. shape
        (C, T, H, W) for SpikingMNIST or (T, features) for a flat loader.
        """
        x, y = self.dataloader.sample(trail=trail)

        if x.dim() < 3:
            raise ValueError(
                f"Spike raster requires a temporal dimension but got shape {tuple(x.shape)}. "
                "Use an ImageToSpikeConverter-based loader."
            )

        # Collapse to (T, features) regardless of original shape.
        if x.dim() == 4:
            # (C, T, H, W) → (T, C*H*W)
            c, t, h, w = x.shape
            flat = x.permute(1, 0, 2, 3).reshape(t, c * h * w)
        elif x.dim() == 3:
            # (C, T, features) or (T, H, W) → take first channel or flatten spatial
            flat = x[0] if x.shape[0] < x.shape[1] else x.reshape(x.shape[0], -1)
        else:
            flat = x  # already (T, features)

        sample = flat.numpy()
        t, n = sample.shape

        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(max(6, t / 6), min(6, n / 20 + 2)))

        ax.imshow(
            sample.T,
            aspect="auto",
            cmap="binary",
            interpolation="nearest",
            origin="lower",
        )
        ax.set_xlabel("Time step")
        ax.set_ylabel("Feature index")
        label = y.item() if hasattr(y, "item") else y
        ax.set_title(f"Spike raster  (label: {label}, {t} steps, {n} features)")

        if created:
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------

    def plot_histogram(self, trail: str = "train", ax: plt.Axes | None = None) -> None:
        """Distribution of raw pixel / spike values across one batch."""
        x, _ = self.dataloader.sample(trail=trail)
        values = x.detach().cpu().flatten().numpy()

        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(6, 3))

        ax.hist(values, bins=60, color="steelblue", alpha=0.8, edgecolor="none")
        ax.axvline(values.mean(), color="tomato", linewidth=1.5, label=f"mean={values.mean():.3f}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.set_title(f"Value distribution ({trail} set, 1 sample)")
        ax.legend(fontsize=8)

        if created:
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_image(self, x) -> np.ndarray:
        """Convert any sample tensor to a plottable 2-D array."""
        x = x.detach().cpu()
        if x.dim() == 1:
            side = int(x.shape[0] ** 0.5)
            return x.reshape(side, side).numpy()
        if x.dim() == 2:
            return x.numpy()
        if x.dim() == 3:
            # (C, H, W): grayscale or RGB
            if x.shape[0] == 1:
                return x[0].numpy()
            return x.permute(1, 2, 0).numpy()
        if x.dim() == 4:
            # (C, T, H, W): spike train — accumulate over time
            return x[0].sum(dim=0).numpy()  # sum T, take first channel
        raise ValueError(f"Cannot convert shape {tuple(x.shape)} to image.")
