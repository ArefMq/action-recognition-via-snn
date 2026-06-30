from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch

if TYPE_CHECKING:
    from spikenet.network.network import Network

K = 30  # max neurons shown in the membrane potential row


def plot_network_activity(
    network: Network,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    """Per-epoch network activity: membrane potential traces (top) and spike rasters (bottom).

    Top row — membrane potential of up to K randomly sampled neurons per layer.
    Bottom row — full spike raster (all neurons x time), anti-aliased for display.

    Call this as an epoch_callback during Network.train():
        net.train(data, epoch_callback=lambda ep, m: plot_network_activity(net, ep, m))
    """
    from spikenet.layers.spiking_base import SpikingNeuron

    active = [
        (idx, layer)
        for idx, layer in enumerate(network._layers)
        if isinstance(layer, SpikingNeuron) and layer._raw_spike_rec is not None and layer._mem_rec is not None
    ]

    if not active:
        return

    # ---- print per-layer firing rates ----
    print(f"\n  [activity] epoch {epoch}")
    for _, layer in active:
        assert layer._raw_spike_rec is not None
        rate = layer._raw_spike_rec.float().mean().item() * 100
        print(f"    {layer.name}: {rate:.1f}% avg firing")
    print("")

    n = len(active)
    fig, axes = plt.subplots(2, n, figsize=(max(4 * n, 6), 7))
    if n == 1:
        axes = axes.reshape(2, 1)

    rng = torch.Generator()

    for col, (layer_idx, layer) in enumerate(active):
        label = f"#{layer_idx} {layer.name}"

        # ---- top: membrane potential ----
        assert layer._mem_rec is not None and layer._raw_spike_rec is not None
        mem = layer._mem_rec[0].detach().cpu()  # (time, *features)
        if mem.dim() > 2:
            mem = mem.reshape(mem.shape[0], -1)  # (time, neurons)
        n_neurons = mem.shape[1]

        if n_neurons > K:
            sel = torch.randperm(n_neurons, generator=rng)[:K]
            mem_plot = mem[:, sel].numpy()
            note = f"{K} of {n_neurons} neurons"
        else:
            mem_plot = mem.numpy()
            note = f"{n_neurons} neurons"

        ax_mem = axes[0, col]
        t = range(mem_plot.shape[0])
        for i in range(mem_plot.shape[1]):
            ax_mem.plot(t, mem_plot[:, i], alpha=0.5, linewidth=0.7)
        ax_mem.set_title(f"{label}\n{note}", fontsize=8)
        ax_mem.set_xlabel("t", fontsize=7)
        ax_mem.set_ylabel("V", fontsize=7)
        ax_mem.tick_params(labelsize=6)
        ax_mem.grid(True, alpha=0.2)

        # ---- bottom: spike raster (all neurons, anti-aliased) ----
        spikes = layer._raw_spike_rec[0].detach().cpu()  # (time, *features)
        if spikes.dim() > 2:
            spikes = spikes.reshape(spikes.shape[0], -1)
        raster = spikes.numpy().T  # (neurons, time)

        ax_raster = axes[1, col]
        ax_raster.imshow(
            raster,
            aspect="auto",
            cmap="binary",
            interpolation="antialiased",
            vmin=0,
            vmax=1,
        )
        ax_raster.set_title("spike raster", fontsize=8)
        ax_raster.set_xlabel("t", fontsize=7)
        ax_raster.set_ylabel("neuron #", fontsize=7)
        ax_raster.tick_params(labelsize=6)

        legend_patches = [
            mpatches.Patch(facecolor="black", label="spike"),
            mpatches.Patch(facecolor="white", edgecolor="grey", label="silent"),
        ]
        ax_raster.legend(
            handles=legend_patches,
            loc="lower right",
            fontsize=6,
            framealpha=0.7,
            handlelength=1.0,
            handleheight=0.8,
        )

    fig.suptitle(
        f"Network activity — epoch {epoch}  |  loss {metrics['loss']:.4f}  acc {metrics['accuracy']:.4f}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()
