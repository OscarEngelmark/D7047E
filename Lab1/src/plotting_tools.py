from typing import List, Optional, Literal, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline as _backend_inline

# Render inline plots as SVG so all edges stay crisp regardless of screen DPI.
_backend_inline.set_matplotlib_formats("svg")


def plot_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Collect predictions from `loader` and plot a confusion matrix.

    Works for binary and multiclass classification. Runs on the same
    device as the model and accumulates via TorchMetrics for correctness
    across batches.

    Parameters
    ----------
    model       : trained model in eval-ready state
    loader      : DataLoader for the split to evaluate (e.g. test_loader)
    num_classes : total number of output classes
    class_names : optional list of human-readable labels, e.g. ['Neg', 'Pos']
                  falls back to '0', '1', ... if not provided
    title       : plot title

    Returns
    -------
    cm : (num_classes, num_classes) confusion matrix tensor (counts)
    """
    device = next(model.parameters()).device
    task = cast(Literal["binary", "multiclass", "multilabel"],
                "binary" if num_classes == 2 else "multiclass")
    metric = ConfusionMatrix(task=task, num_classes=num_classes).to(device)

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(dim=1)
            metric.update(preds, labels)

    cm = metric.compute()

    # --- Plot ---
    labels_ = class_names if class_names else [str(i) for i in range(num_classes)]
    fig, ax = plt.subplots(figsize=(max(4, num_classes), max(4, num_classes)))
    im = ax.imshow(cm.cpu().numpy(), cmap="Blues")

    # Attach the colorbar to `ax` with a fixed fractional width so it never
    # bleeds into the title, regardless of how tall or wide the grid is.
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(labels_, rotation=45, ha="right")
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(labels_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    # `y=1.02` lifts the title clear of the axes box so tight_layout
    # has an accurate bounding box to work with.
    ax.set_title(title, pad=10, y=1.02)

    # Annotate each cell with its count
    cm_max = cm.max().item()
    for i in range(num_classes):
        for j in range(num_classes):
            count = cm[i, j].item()
            color = "white" if count > cm_max * 0.6 else "black"
            ax.text(j, i, str(int(count)), ha="center", va="center", color=color)

    # `rect` reserves a small top margin so the raised title is never clipped.
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()