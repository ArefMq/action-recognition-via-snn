from __future__ import annotations

import matplotlib.pyplot as plt


def plot_training_history(
    history: list[dict[str, float]],
    test_metrics: dict[str, float] | None = None,
) -> None:
    """Plot loss and accuracy curves from the list returned by Network.train().

    Args:
        history: List of per-epoch dicts with keys 'epoch', 'loss', 'accuracy'.
        test_metrics: Optional dict with 'loss' and 'accuracy' from Network.test(),
                      drawn as horizontal reference lines.
    """
    if not history:
        raise ValueError("History is empty — call net.train(data) first.")

    epochs = [m["epoch"] for m in history]
    losses = [m["loss"] for m in history]
    accuracies = [m["accuracy"] for m in history]

    _, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, losses, "o-", color="steelblue", markersize=4, linewidth=1.5, label="train")
    if test_metrics is not None:
        ax_loss.axhline(
            test_metrics["loss"],
            color="tomato",
            linewidth=1.5,
            linestyle="--",
            label=f"test {test_metrics['loss']:.4f}",
        )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, accuracies, "o-", color="seagreen", markersize=4, linewidth=1.5, label="train")
    if test_metrics is not None:
        ax_acc.axhline(
            test_metrics["accuracy"],
            color="tomato",
            linewidth=1.5,
            linestyle="--",
            label=f"test {test_metrics['accuracy']:.4f}",
        )
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title("Accuracy")
    ax_acc.legend(fontsize=9)
    ax_acc.grid(True, alpha=0.3)

    plt.suptitle("Training history", y=1.02)
    plt.tight_layout()
    plt.show()
