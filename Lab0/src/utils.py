"""
utils.py — Shared training helpers.

Exports
-------
train   : one training epoch
validate: one evaluation epoch
fit     : full training loop with wandb logging and best-checkpoint restore
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader


def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """Run one full pass over `loader` in training mode.

    Returns (avg_loss, accuracy_%).
    """
    device = next(model.parameters()).device
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct      += (outputs.argmax(dim=1) == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """Run one full pass over `loader` in evaluation mode.

    Returns (avg_loss, accuracy_%).
    """
    device = next(model.parameters()).device
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            correct      += (outputs.argmax(dim=1) == labels).sum().item()
            total        += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def fit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    wandb_kwargs: Dict[str, Any],
    log: bool = True
) -> Dict[str, List[float]]:
    """Train `model` for `num_epochs`, validating after every epoch.

    Initialises and closes a wandb run for the duration of training.
    Saves the weights that achieved the best validation accuracy and
    restores them at the end.

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """

    if not log:
        wandb_kwargs = {**wandb_kwargs, "mode": "disabled"}

    with wandb.init(**wandb_kwargs):

        best_val_acc = 0.0
        best_state   = None
        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion)
            val_loss,   val_acc   = validate(model, val_loader, criterion)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(
                f"Epoch {epoch:>{len(str(num_epochs))}}/{num_epochs} | "
                f"train loss {train_loss:.4f}, train acc {train_acc:.2f}% | "
                f"val loss {val_loss:.4f}, val acc {val_acc:.2f}%"
            )

            wandb.log({
                "Training Loss":       train_loss,
                "Training Accuracy":   train_acc,
                "Validation Loss":     val_loss,
                "Validation Accuracy": val_acc,
            })

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best weights (val acc {best_val_acc:.2f}%)")

    return history