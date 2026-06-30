from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.axes import Axes


class Metrics:
    """Accumulates training history across one or more training phases.

    Created by Network.train(). Supports += to concatenate phases, adding a
    vertical separator line on plots to mark where each phase began.

    Usage::

        m = net.train(data, epochs=3)
        m += net.train(data, epochs=7)   # continues from where the first left off
        m.plot()
        m.print()
    """

    def __init__(
        self,
        epoch_history: list[dict[str, float]] | None = None,
        batch_history: list[dict[str, float]] | None = None,
        test_metrics: dict[str, float] | None = None,
        _phase_boundaries: list[float] | None = None,
    ) -> None:
        self._epochs: list[dict[str, float]] = epoch_history or []
        self._batches: list[dict[str, float]] = batch_history or []
        self._test = test_metrics
        self._boundaries: list[float] = _phase_boundaries or []

    # ---- accessors ----

    @property
    def loss(self) -> float:
        return self._epochs[-1]["loss"]

    @property
    def accuracy(self) -> float:
        return self._epochs[-1]["accuracy"]

    @property
    def test(self) -> dict[str, float] | None:
        return self._test

    @property
    def epoch_history(self) -> list[dict[str, float]]:
        return self._epochs

    @property
    def batch_history(self) -> list[dict[str, float]]:
        return self._batches

    # ---- combination ----

    def __iadd__(self, other: Metrics) -> Metrics:
        if not self._epochs:
            self._epochs = list(other._epochs)
            self._batches = list(other._batches)
            self._test = other._test
            self._boundaries = list(other._boundaries)
            return self

        offset = self._epochs[-1]["epoch"]
        self._boundaries.append(offset)

        for e in other._epochs:
            if e["epoch"] == 0.0:
                continue  # pre-training baseline of the new phase duplicates the last point of self
            self._epochs.append({**e, "epoch": e["epoch"] + offset})
        for b in other._batches:
            if b["epoch_frac"] == 0.0:
                continue
            self._batches.append({**b, "epoch_frac": b["epoch_frac"] + offset})
        for b in other._boundaries:
            self._boundaries.append(b + offset)

        if other._test is not None:
            self._test = other._test
        return self

    # ---- display ----

    def print(self) -> None:
        """Print a per-epoch summary table using rich."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Training history", show_header=True, header_style="bold")
        table.add_column("Epoch", justify="right", style="dim")
        table.add_column("Loss", justify="right")
        table.add_column("Accuracy", justify="right")

        boundary_epochs = {int(b) for b in self._boundaries}
        for e in self._epochs:
            ep = int(e["epoch"])
            table.add_row(str(ep), f"{e['loss']:.4f}", f"{e['accuracy']:.2%}")
            if ep in boundary_epochs:
                table.add_section()

        console.print(table)
        if self._test:
            console.print(
                f"Test — loss: [yellow]{self._test['loss']:.4f}[/yellow]  "
                f"acc: [green]{self._test['accuracy']:.2%}[/green]"
            )

    def plot(self) -> None:
        """Plot loss and accuracy with per-batch oscillation and epoch averages.

        Each subplot shows:
          - thin transparent line: per-batch values (shows oscillation within epochs)
          - thick line: per-epoch average
          - dashed vertical lines: phase boundaries (from +=)
          - dashed horizontal line: test reference (if available)
        """
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

        epoch_xs = [e["epoch"] for e in self._epochs]
        batch_xs = [b["epoch_frac"] for b in self._batches]

        def _add_separators(ax: Axes) -> None:
            for i, b in enumerate(self._boundaries):
                ax.axvline(
                    b + 0.5,
                    color="gray",
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.7,
                    label="phase boundary" if i == 0 else None,
                )

        # Loss
        ax_loss.plot(
            batch_xs,
            [b["loss"] for b in self._batches],
            alpha=0.15,
            linewidth=0.6,
            color="steelblue",
        )
        ax_loss.plot(
            epoch_xs,
            [e["loss"] for e in self._epochs],
            linewidth=2.0,
            color="steelblue",
            label="epoch avg",
        )
        if self._test:
            ax_loss.axhline(
                self._test["loss"],
                color="tomato",
                linestyle="--",
                linewidth=1.2,
                label=f"test {self._test['loss']:.4f}",
            )
        _add_separators(ax_loss)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Loss")
        ax_loss.legend(fontsize=7)
        ax_loss.grid(True, alpha=0.2)

        # Accuracy
        ax_acc.plot(
            batch_xs,
            [b["accuracy"] for b in self._batches],
            alpha=0.15,
            linewidth=0.6,
            color="seagreen",
        )
        ax_acc.plot(
            epoch_xs,
            [e["accuracy"] for e in self._epochs],
            linewidth=2.0,
            color="seagreen",
            label="epoch avg",
        )
        if self._test:
            ax_acc.axhline(
                self._test["accuracy"],
                color="tomato",
                linestyle="--",
                linewidth=1.2,
                label=f"test {self._test['accuracy']:.2%}",
            )
        _add_separators(ax_acc)
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title("Accuracy")
        ax_acc.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax_acc.legend(fontsize=7)
        ax_acc.grid(True, alpha=0.2)

        fig.suptitle("Training history", fontsize=11)
        plt.tight_layout()
        plt.show()
