"""
utils.py — Shared training helpers.

Exports
-------
device_check      : detect and return the best available torch.device
stratified_split  : stratified train/val/test split for any labelled DataFrame
make_loaders      : split a dataset into train/val loaders and wrap the test set
train             : one training epoch
validate          : one evaluation epoch
fit               : full training loop with wandb logging and best-checkpoint restore
evaluate          : evaluate a trained model on the test loader and print a summary
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import sys, platform
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def device_check() -> torch.device:
    """Detect the best available device and print a summary of the environment.

    Prefers CUDA, then MPS (Apple Silicon), then falls back to CPU.

    Returns
    -------
    torch.device
        The selected device (cuda, mps, or cpu).
    """

    print(f"PyTorch: {torch.__version__} | "
          f"Python: {sys.version.split()[0]} | "
          f"OS: {platform.system()} {platform.release()}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPUs: {gpu_count} x {gpu_name} ({total_mem:.1f} GB)")
        print(f"CUDA: {torch.version.cuda} | "
              f"cuDNN: {torch.backends.cudnn.version()}")
        printout = f"Using cuda / {gpu_name}"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        printout = "Using mps (Apple Silicon)"
    else:
        device = torch.device("cpu")
        printout = f"Using cpu"

    print(printout)
    return device

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def stratified_split(
    data: "pd.DataFrame",
    label_col: str = "Class",
    test_size: float = 0.10,
    val_size: float = 0.10,
    seed: int = 1,
) -> Dict[str, pd.DataFrame]:
    """Split a DataFrame into stratified train / val / test subsets.
 
    Performs two successive stratified splits: first carves out test_size
    as the test set, then splits the remainder with val_size into val and
    train. With defaults this yields an 81 / 9 / 10 split. Note that
    val_size is a fraction of the train+val pool, not the full dataset.
 
    Returns a dict with keys "train", "val", "test", each a
    DataFrame with a reset index.
    """

    trainval, test = train_test_split(
        data,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=data[label_col],
    )
    train, val = train_test_split(
        trainval,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=trainval[label_col],
    )

    splits = {
        "train": train.reset_index(drop=True),
        "val":   val.reset_index(drop=True),
        "test":  test.reset_index(drop=True),
    }

    total = len(data)
    print(f"Stratified split — total: {total:,}  |  seed={seed}")
    for name, df in splits.items():
        pct    = 100 * len(df) / total
        counts = df[label_col].value_counts().sort_index()
        dist   = "  ".join(f"{k}={v:,}" for k, v in counts.items())
        print(f"  {name:5s}: {len(df):>7,}  ({pct:4.1f} %)  [{dist}]")

    return splits

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

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
    log: bool = True,
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

        best_val_loss = float('inf')
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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
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
        print(f"\nRestored best weights (val loss {best_val_loss:.4f})")

    return history

def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    label: str = "Test",
) -> Tuple[float, float]:
    """Evaluate `model` on `test_loader` and print a one-line summary.

    Parameters
    ----------
    model       : trained model
    test_loader : DataLoader for the held-out test set
    criterion   : loss function
    label       : name printed in the summary line (e.g. the experiment name)

    Returns
    -------
    (test_loss, test_acc_%)
    """
    test_loss, test_acc = validate(model, test_loader, criterion)
    print(f"[{label}] Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")
    return test_loss, test_acc