from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from spikenet.layers.spiking_base import SpikingNeuron


class LayerVisualizer:
    """Visualisation for a single spiking layer.

    Handles SpikingDenseLayer (2-D weights, 3-D spike/mem records) and
    SpikingConv2D (5-D weights, 5-D spike/mem records).
    """

    def __init__(self, layer: SpikingNeuron) -> None:
        self.layer = layer

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def plot_weights(self, ax: plt.Axes | None = None) -> None:
        """Heatmap of the weight matrix (dense) or filter grid (conv)."""
        if self.layer.w is None:
            raise RuntimeError(
                f"'{self.layer.name}' has no learnable weights. "
                "Call initialize_parameters() first, or this layer type has no weights."
            )

        w = self.layer.w.detach().cpu()

        if w.dim() == 2:
            # SpikingDenseLayer: (in_features, out_features)
            created = ax is None
            if created:
                fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(w.numpy(), aspect="auto", cmap="RdBu_r")
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{self.layer.name}  [{w.shape[0]} x {w.shape[1]}]")
            ax.set_xlabel("Output features")
            ax.set_ylabel("Input features")
            if created:
                plt.tight_layout()
                plt.show()

        elif w.dim() == 5:
            # SpikingConv2D: (out_ch, in_ch, 1, kH, kW)
            # Average over input channels to get one 2-D kernel per filter.
            out_ch, in_ch, _, kh, kw = w.shape
            kernels = w[:, :, 0, :, :].mean(dim=1)  # (out_ch, kH, kW)
            n_cols = min(8, out_ch)
            n_rows = (out_ch + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5 + 0.4))
            flat = np.array(axes).flatten() if out_ch > 1 else [axes]
            vmax = float(kernels.abs().max())
            for a, k in zip(flat, kernels, strict=False):
                a.imshow(k.numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                a.axis("off")
            for a in flat[out_ch:]:
                a.axis("off")
            fig.suptitle(f"{self.layer.name}  [out={out_ch}, in={in_ch}, {kh}x{kw}]")
            plt.tight_layout()
            plt.show()

        else:
            raise NotImplementedError(f"Cannot plot weights of shape {tuple(w.shape)}")

    # ------------------------------------------------------------------
    # Spike activity
    # ------------------------------------------------------------------

    def plot_activity(self, ax: plt.Axes | None = None) -> None:
        """Spike raster (dense) or mean-activity grid (conv) from the last forward pass."""
        if self.layer._spike_rec is None:
            raise RuntimeError(f"'{self.layer.name}' has no spike record. Run a forward pass first.")

        spk = self.layer._spike_rec.detach().cpu()

        if spk.dim() == 3:
            # Dense (no time reduction): (batch, time, neurons)
            sample = spk[0].numpy()  # (time, neurons)
            t, n = sample.shape
            created = ax is None
            if created:
                fig, ax = plt.subplots(figsize=(max(6, t / 6), min(8, n / 8 + 1)))
            ax.imshow(
                sample.T,
                aspect="auto",
                cmap="binary",
                interpolation="nearest",
                origin="lower",
            )
            ax.set_xlabel("Time step")
            ax.set_ylabel("Neuron")
            ax.set_title(f"{self.layer.name} — spike raster  [{t} steps, {n} neurons]")
            if created:
                plt.tight_layout()
                plt.show()

        elif spk.dim() == 5:
            # Conv (no time reduction): (batch, channels, time, h, w)
            # Show mean activity (over time) per output channel.
            activity = spk[0].mean(dim=1)  # (channels, h, w)
            n_ch = min(activity.shape[0], 16)
            n_cols = min(8, n_ch)
            n_rows = (n_ch + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5 + 0.4))
            flat = np.array(axes).flatten() if n_ch > 1 else [axes]
            for i, (a, ch_map) in enumerate(zip(flat[:n_ch], activity[:n_ch], strict=False)):
                a.imshow(ch_map.numpy(), cmap="hot", vmin=0, vmax=1)
                a.set_title(f"ch {i}", fontsize=7)
                a.axis("off")
            for a in flat[n_ch:]:
                a.axis("off")
            fig.suptitle(f"{self.layer.name} — mean spike activity per channel")
            plt.tight_layout()
            plt.show()

        elif spk.dim() == 2:
            # Time-reduced output layer: (batch, features)
            sample = spk[0].numpy()
            created = ax is None
            if created:
                fig, ax = plt.subplots(figsize=(max(4, len(sample) / 5), 3))
            ax.bar(range(len(sample)), sample, color="steelblue", width=0.8)
            ax.set_xlabel("Output neuron")
            ax.set_ylabel("Score")
            ax.set_title(f"{self.layer.name} — output scores")
            if created:
                plt.tight_layout()
                plt.show()

        else:
            raise NotImplementedError(f"Cannot plot activity for spike_rec shape {tuple(spk.shape)}")

    # ------------------------------------------------------------------
    # Membrane potential
    # ------------------------------------------------------------------

    def plot_membrane(self, ax: plt.Axes | None = None) -> None:
        """Membrane potential trace from the last forward pass."""
        if self.layer._mem_rec is None:
            raise RuntimeError(
                f"'{self.layer.name}' has no membrane potential record. "
                "This layer may not track membrane dynamics (e.g. pooling), "
                "or no forward pass has been run yet."
            )

        mem = self.layer._mem_rec.detach().cpu()

        if mem.dim() == 3:
            # Dense: (batch, time, neurons)
            sample = mem[0].numpy()  # (time, neurons)
            t, n = sample.shape
            created = ax is None
            if created:
                fig, ax = plt.subplots(figsize=(max(6, t / 6), 3))
            if n <= 10:
                for i in range(n):
                    ax.plot(sample[:, i], alpha=0.7, linewidth=1, label=f"n{i}")
                ax.legend(fontsize=7, loc="upper right", ncol=2)
            else:
                im = ax.imshow(
                    sample.T,
                    aspect="auto",
                    cmap="plasma",
                    interpolation="nearest",
                    origin="lower",
                )
                plt.colorbar(im, ax=ax)
                ax.set_ylabel("Neuron")
            ax.set_xlabel("Time step")
            ax.set_title(f"{self.layer.name} — membrane potential  [{n} neurons]")
            if created:
                plt.tight_layout()
                plt.show()

        elif mem.dim() == 5:
            # Conv: (batch, channels, time, h, w)
            # Spatial mean per channel → (channels, time)
            sample = mem[0].mean(dim=(2, 3)).numpy()
            n_ch = min(sample.shape[0], 16)
            created = ax is None
            if created:
                fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(sample[:n_ch].T, alpha=0.7, linewidth=1)
            ax.set_xlabel("Time step")
            ax.set_ylabel("Mean membrane potential")
            ax.set_title(f"{self.layer.name} — spatially-averaged membrane  [first {n_ch} channels]")
            if created:
                plt.tight_layout()
                plt.show()

        else:
            raise NotImplementedError(f"Cannot plot membrane for shape {tuple(mem.shape)}")
