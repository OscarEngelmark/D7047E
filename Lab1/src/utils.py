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
        print(f"  {name:5s}: {len(df):>10,}  ({pct:4.1f} %)  [{dist}]")

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
    
    Uses AMP automatically when CUDA is available.
    """
    device  = next(model.parameters()).device
    cuda_available    = torch.cuda.is_available()
    scaler  = torch.amp.GradScaler('cuda') if cuda else None  # type: ignore[attr-defined]

    running_loss = 0.0
    correct      = 0
    total        = 0

    model.train()

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=cuda_available):
            outputs = model(inputs)
            loss    = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
    cuda_available = device.type == 'cuda'
    
    running_loss = 0.0
    correct      = 0
    total        = 0

    model.eval()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.autocast(device_type=device.type, enabled=cuda_available):
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
    patience: Optional[int] = None,
    min_delta: float = 1e-4,
) -> Dict[str, List[float]]:
    """Train `model` for up to `num_epochs`, validating after every epoch.

    Initialises and closes a wandb run for the duration of training.
    Saves the weights that achieved the best validation loss and restores
    them at the end.

    Prints a header row followed by one row per epoch for the first 5 epochs,
    then every 5 epochs thereafter. If early stopping triggers on an unprinted
    epoch, that epoch's row is printed before the stopping message.

    Parameters
    ----------
    model        : the network to train; modified in-place
    optimizer    : optimiser (e.g. Adam) already constructed for `model`
    criterion    : loss function (e.g. CrossEntropyLoss)
    train_loader : DataLoader for the training split
    val_loader   : DataLoader for the validation split
    num_epochs   : maximum number of epochs to train for
    wandb_kwargs : keyword arguments forwarded to wandb.init()
    log          : if False, disables wandb logging entirely
    patience     : number of consecutive epochs without improvement before
                   stopping early; None disables early stopping
    min_delta    : minimum absolute decrease in validation loss to count as
                   an improvement (filters out numerical noise)

    Returns
    -------
    history : dict with keys 'Training Loss', 'Validation Loss',
              'Training Accuracy', 'Validation Accuracy', each mapping to
              a list of per-epoch values
    """

    if not log:
        wandb_kwargs = {**wandb_kwargs, "mode": "disabled"}

    if torch.cuda.is_available():
        model = torch.compile(model)   # type: ignore[assignment]

    best_val_loss = float('inf')
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0
    history: Dict[str, List[float]] = {
        "Training Loss": [], "Validation Loss": [],
        "Training Accuracy":  [], "Validation Accuracy":  [],
    }
    
    w = len(str(num_epochs))
    epoch_col_w = max(2 * w + 1, 5)
    col_w = 10
    header = (
        f"{'Epoch':>{epoch_col_w}} | "
        f"{'Train Loss':>{col_w}} | "
        f"{'Train Acc':>{col_w}} | "
        f"{'Val Loss':>{col_w}} | "
        f"{'Val Acc':>{col_w}}"
    )

    with wandb.init(**wandb_kwargs):

        print(header)

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion)
            val_loss,   val_acc   = validate(model, val_loader, criterion)

            history["Training Loss"].append(train_loss)
            history["Validation Loss"].append(val_loss)
            history["Training Accuracy"].append(train_acc)
            history["Validation Accuracy"].append(val_acc)

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            early_stop = patience is not None and epochs_no_improve >= patience
            
            if epoch <= 5 or epoch % 5 == 0 or early_stop:  # Print first 5, every 5th, and the stopping epoch
                print(
                    f"{epoch:>{w}}/{num_epochs:>{w}} | "
                    f"{train_loss:>{col_w}.4f} | "
                    f"{train_acc:>{col_w - 1}.2f}% | "
                    f"{val_loss:>{col_w}.4f} | "
                    f"{val_acc:>{col_w - 1}.2f}%"
                )

            wandb.log({k: v[-1] for k, v in history.items()})

            if early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch} (no improvement for {patience} epochs)")
                break

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