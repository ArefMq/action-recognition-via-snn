import math
from collections.abc import Callable, Iterator

import torch
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from torch import Tensor
from typing_extensions import Self

from spikenet.constants import DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_MOMENTUM, DEFAULT_WEIGHT_DECAY
from spikenet.data import DataLoader
from spikenet.layers.flattening import Flatten
from spikenet.layers.neuron_base import NeuronBase
from spikenet.network.criterion import Criterion
from spikenet.network.metrics import Metrics
from spikenet.visual.mixins import NetworkPlottable


class Network(torch.nn.Module, NetworkPlottable):
    def __init__(
        self,
        *args,
        name: str = "SpikingNetwork",
        device: torch.device | None = None,
        optimizer_generator: torch.optim.Optimizer = torch.optim.SGD,
        optimizer: torch.optim.Optimizer | None = None,
        loss_fn: torch.nn.Module = torch.nn.NLLLoss,
        encoding: torch.nn.Module = None,
        epochs: int = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        momentum: float = DEFAULT_MOMENTUM,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.criterion = Criterion(
            optimizer_generator=optimizer_generator,
            optimizer=optimizer,
            loss_fn=loss_fn,
            encoding=encoding or torch.nn.LogSoftmax(dim=1),
            epochs=epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.name = name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._is_compiled: bool = False
        self._layers: list[torch.nn.Module] = []

    # ~~~~~~~~~~~~~~~~~~~~~ Constructing Network ~~~~~~~~~~~~~~~~~~~~~
    def add_layer(
        self,
        module: torch.nn.Module,
    ) -> Self:
        self._is_compiled = False
        self._layers.append(module)
        return self

    def __iadd__(self, other: "torch.nn.Module | Network") -> Self:
        if isinstance(other, Network):
            for layer in other._layers:
                self.add_layer(layer)
        else:
            self.add_layer(other)
        return self

    def __add__(self, other: "Network") -> "Network":
        new_net = Network(name=f"{self.name}+{other.name}")
        for layer in self._layers:
            new_net.add_layer(layer)
        for layer in other._layers:
            new_net.add_layer(layer)
        return new_net

    def fit(self, dataloader: DataLoader, **kwargs) -> Self:
        input_shape, output_shape = dataloader.shape
        self.compiled(input_features=input_shape[1], output_features=output_shape[1])
        assert self.validate_layers(), "Network layers are not properly configured"
        self.train(dataloader, **kwargs)
        self.test(dataloader)
        return self

    def train(
        self,
        dataloader: DataLoader | bool = True,
        epochs: int | None = None,
        learning_rate: float | None = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        epoch_callback: Callable[[int, dict[str, float]], None] | None = None,
        max_grad_norm: float | None = 1.0,
        run_test: bool = True,
    ) -> Metrics:
        if isinstance(dataloader, bool):
            return super().train(dataloader)
        super().train()
        self.to(self.device)

        # Dry run to trigger lazy parameter initialization in layers whose
        # in_features can't be known at compile time (e.g. SpikingDenseLayer
        # after a Flatten, where the actual flattened size depends on spatial dims).
        for x, _ in dataloader("train"):
            with torch.no_grad():
                self(x[:1].to(self.device))
            break

        num_epochs = epochs if epochs is not None else self.criterion.epochs
        optimizer = self.criterion.get_optim(self, lr=learning_rate)
        loss_fn = self.criterion.get_loss_fn(self)

        num_batches_per_epoch = math.ceil(dataloader.len("train") / dataloader.batch_size)

        # Epoch-0 baseline: eval pass before any weight updates.
        super().eval()
        with torch.no_grad():
            init_loss, init_correct, init_total, init_batches = 0.0, 0, 0, 0
            for x, y in dataloader("train"):
                x, y = x.to(self.device), y.to(self.device)
                encoded = self.criterion.encoding(self(x))
                init_loss += loss_fn(encoded, y).item()
                init_correct += (encoded.argmax(dim=1) == y).sum().item()
                init_total += y.size(0)
                init_batches += 1
        super().train()
        _init = {"epoch": 0.0, "loss": init_loss / init_batches, "accuracy": init_correct / init_total}
        epoch_history: list[dict[str, float]] = [_init]
        batch_history: list[dict[str, float]] = [
            {"epoch_frac": 0.0, **{k: v for k, v in _init.items() if k != "epoch"}}
        ]

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            num_batches = 0

            with Progress(
                SpinnerColumn(),
                TextColumn(" [bold cyan]Epoch {task.fields[ep]}/{task.fields[total_ep]}[/bold cyan]"),
                BarColumn(bar_width=40),
                MofNCompleteColumn(),
                TextColumn("  [yellow]loss={task.fields[loss]:.4f}[/yellow]"),
                TextColumn("[green]acc={task.fields[acc]:.1%}[/green]"),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    "",
                    total=num_batches_per_epoch,
                    ep=epoch + 1,
                    total_ep=num_epochs,
                    loss=0.0,
                    acc=0.0,
                )

                for x, y in dataloader("train"):
                    x = x.to(self.device)
                    y = y.to(self.device)

                    optimizer.zero_grad()
                    output = self(x)
                    encoded = self.criterion.encoding(output)
                    loss = loss_fn(encoded, y)
                    loss.backward()
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                    optimizer.step()
                    self.clamp()

                    epoch_loss += loss.item()
                    predictions = encoded.argmax(dim=1)
                    batch_correct = (predictions == y).sum().item()
                    correct += batch_correct
                    total += y.size(0)
                    num_batches += 1

                    batch_history.append(
                        {
                            "epoch_frac": epoch + num_batches / num_batches_per_epoch,
                            "loss": loss.item(),
                            "accuracy": batch_correct / y.size(0),
                        }
                    )
                    progress.update(task, advance=1, loss=epoch_loss / num_batches, acc=correct / total)

            if lr_scheduler is not None:
                lr_scheduler.step()

            epoch_metrics = {
                "epoch": float(epoch + 1),
                "loss": epoch_loss / num_batches,
                "accuracy": correct / total,
            }
            epoch_history.append(epoch_metrics)
            print(
                f"Epoch [{epoch + 1}/{num_epochs}]  loss={epoch_metrics['loss']:.4f}  "
                f"acc={epoch_metrics['accuracy']:.4f}"
            )
            if epoch_callback is not None:
                epoch_callback(epoch + 1, epoch_metrics)

        test_metrics = self.test(dataloader, verbose=False) if run_test else None
        return Metrics(epoch_history, batch_history, test_metrics)

    def test(self, dataloader: DataLoader, verbose: bool = True) -> dict[str, float]:
        super().eval()
        self.to(self.device)

        loss_fn = self.criterion.get_loss_fn(self)
        test_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        with torch.no_grad():
            for x, y in dataloader("test"):
                x = x.to(self.device)
                y = y.to(self.device)

                output = self(x)
                encoded = self.criterion.encoding(output)
                loss = loss_fn(encoded, y)

                test_loss += loss.item()
                predictions = encoded.argmax(dim=1)
                correct += (predictions == y).sum().item()
                total += y.size(0)
                num_batches += 1

        metrics = {
            "loss": test_loss / num_batches,
            "accuracy": correct / total,
        }
        if verbose:
            print(f"Test  loss={metrics['loss']:.4f}  acc={metrics['accuracy']:.4f}")
        return metrics

    # ~~~~~~~~~~~~~~~~~~~~~~~ Compile Network ~~~~~~~~~~~~~~~~~~~~~~~~~
    def compiled(
        self,
        input_features: int | None = None,
        output_features: int | None = None,
        data_shape: tuple[int, ...] | None = None,
    ) -> None:
        if self._is_compiled:
            return
        self._populate_features(input_features, output_features)
        self._is_compiled = True

        if data_shape is not None:
            # Propagate spatial dims (h, w) through Flatten and any lazy layers by
            # running a single dummy sample — batch size forced to 1.
            self.to(self.device)
            dummy = torch.ones((1, *data_shape[1:]), device=self.device)
            with torch.no_grad():
                self(dummy)

    def _populate_features(
        self,
        input_features: int | None = None,
        output_features: int | None = None,
    ):
        if isinstance(self._layers[0], NeuronBase) and self._layers[0].in_features is None:
            self._layers[0].in_features = input_features
        if isinstance(self._layers[-1], NeuronBase) and self._layers[-1].out_features is None:
            self._layers[-1].out_features = output_features

        for i, layer in enumerate(self._layers):
            if not isinstance(layer, NeuronBase):
                continue
            if i == 0:
                continue
            # Flatten computes its own out_features during forward() from the actual
            # spatial dims — we can't know it at compile time, so skip it entirely.
            if isinstance(layer, Flatten):
                continue

            if layer.in_features is None and isinstance(self._layers[i - 1], NeuronBase):
                layer.in_features = self._layers[i - 1].out_features
            if layer.out_features is None and i != len(self._layers) - 1:
                layer.out_features = layer.in_features

    def validate_layers(self) -> bool:
        # Flatten is excluded: its out_features are only known after a forward pass.
        # in_features is allowed to be None for layers that follow a Flatten and
        # self-initialise lazily on the first forward pass.
        return all(
            layer.out_features is not None
            for layer in self._layers
            if isinstance(layer, NeuronBase) and not isinstance(layer, Flatten)
        )

    @property
    def is_compiled(self) -> bool:
        return self._is_compiled and self.validate_layers()

    # ~~~~~~~~~~~~~~~~~~~~~~~~ Using Network ~~~~~~~~~~~~~~~~~~~~~~~~~~
    def forward(self, x: Tensor) -> Tensor:
        assert self.is_compiled, "Network must be compiled before forward pass"
        for layer in self._layers:
            x = layer(x)
        return x

    # ~~~~~~~~~~~~~~~~~~~~~~~ Utility Functions ~~~~~~~~~~~~~~~~~~~~~~~
    def initialize_parameters(self) -> None:
        assert self.is_compiled, "Network must be compiled before initializing parameters"
        for layer in self._layers:
            if isinstance(layer, NeuronBase):
                layer.initialize_parameters()

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        for layer in self._layers:
            yield from layer.parameters()

    def clamp(self) -> None:
        for layer in self._layers:
            if isinstance(layer, NeuronBase):
                layer.clamp()

    def dry_run(self, dataloader: DataLoader) -> bool:
        """Forward pass on one batch to check whether initial firing rates are healthy.

        Prints a per-layer table with firing rate and a colour-coded status:
          green  5-50 %  — healthy range
          yellow 50-80 % — slightly high; consider raising b_init_mean
          red    > 80 %  — too high; raise b_init_mean or lower w_init_mean
          red    < 5 %   — too low; lower b_init_mean

        Returns True if every spiking layer is within 5-80 %.
        """
        from rich.console import Console
        from rich.table import Table

        from spikenet.layers.spiking_base import SpikingNeuron

        self.eval()
        self.to(self.device)
        x, _ = next(iter(dataloader("train")))
        with torch.no_grad():
            self(x.to(self.device))
        self.train()

        console = Console()
        table = Table(title="Dry run — initial weight check", show_header=True, header_style="bold")
        table.add_column("Layer", style="dim")
        table.add_column("Shape")
        table.add_column("Firing rate", justify="right")
        table.add_column("Status")

        all_ok = True
        for layer in self._layers:
            if not isinstance(layer, SpikingNeuron) or layer._raw_spike_rec is None:
                continue
            rate = layer._raw_spike_rec.float().mean().item()
            in_f = str(layer.in_features) if layer.in_features is not None else "?"
            out_f = str(layer.out_features) if layer.out_features is not None else "?"

            if rate > 0.80:
                rate_str, status = f"[red]{rate:.1%}[/red]", "[red]✗ too high — increase b_init_mean[/red]"
                all_ok = False
            elif rate < 0.05:
                rate_str, status = f"[red]{rate:.1%}[/red]", "[red]✗ too low  — decrease b_init_mean[/red]"
                all_ok = False
            elif rate > 0.50:
                rate_str, status = f"[yellow]{rate:.1%}[/yellow]", "[yellow]△ slightly high[/yellow]"
            else:
                rate_str, status = f"[green]{rate:.1%}[/green]", "[green]✓[/green]"

            table.add_row(layer.name, f"{in_f} → {out_f}", rate_str, status)

        console.print(table)
        if all_ok:
            console.print("[green]All layers within healthy range (5-80 %).[/green]\n")
        else:
            console.print("[yellow]⚠  Adjust the flagged layers before training.[/yellow]\n")
        return all_ok

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Visualisation ~~~~~~~~~~~~~~~~~~~~~~~~
    def summarise(self) -> None:
        """Print Summary of the network"""
        print(self.summary())

    def summary(self) -> str:
        """Return the summary of the network structure as string"""
        self.compiled()
        result = [
            self._layer_summary_other_layers(layer)
            if not isinstance(layer, NeuronBase)
            else self._layer_summary_spikenet_layer(layer)
            for layer in self._layers
        ]
        result_magnitude = math.ceil(math.log10(len(result))) if len(result) > 0 else 1
        rows = "\n".join(f"{i:0{result_magnitude}d}) {line}" for i, line in enumerate(result))
        sep = "\u2500" * 54
        total_params = sum(p.numel() for p in self.parameters())
        total_line = f"{'Total':<24s} {'':<16s} {total_params:>8,} params  ({len(result)} layers)"
        return f"{rows}\n{sep}\n{total_line}"

    def _layer_summary_spikenet_layer(self, layer: NeuronBase) -> str:
        name = type(layer).__name__
        in_f = str(layer.in_features) if layer.in_features is not None else "?"
        out_f = str(layer.out_features) if layer.out_features is not None else "?"
        shape = f"{in_f} \u2192 {out_f}"
        num_params = sum(p.numel() for p in layer.parameters())
        line = f"{name:<24s} {shape:<16s} {num_params:>8,} params"

        extras: list[str] = []
        if hasattr(layer, "kernel"):
            extras.append(f"kernel={layer.kernel}")
        out_spatial: tuple[int, ...] | None = getattr(layer, "_out_spatial", None)
        if out_spatial is not None:
            extras.append("x".join(str(s) for s in out_spatial))
        if hasattr(layer, "time_reduction_fn") and layer.time_reduction_fn.__name__ != "no_time_reduction":
            extras.append(f"reduction={layer.time_reduction_fn.__name__}")
        if extras:
            line += "   " + ", ".join(extras)
        return line

    def _layer_summary_other_layers(self, layer: torch.nn.Module) -> str:
        name = layer.__class__.__name__
        num_params = sum(p.numel() for p in layer.parameters())
        return f"{name:<24s} {'':16s} {num_params:>8,} params"
