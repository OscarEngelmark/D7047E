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

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
import pandas as pd
import numpy as np
import sys, platform
import joblib
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score


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
# Model definition
# ---------------------------------------------------------------------------

_ACTIVATIONS: Dict[str, type] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


class SentimentANN(nn.Module):
    """Feedforward classifier with BatchNorm and Dropout for sentiment analysis."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int = 2,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                activation,
                nn.Dropout(dropout),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Run persistence
# ---------------------------------------------------------------------------

def save_ann_run(
    out_dir: str | Path,
    model: nn.Module,
    vectorizer: Any,
    svd: Any,
    input_dim: int,
    hidden_dims: List[int],
    num_classes: int,
    dropout: float,
    activation: str = "relu",
) -> Path:
    """Save a complete ANN pipeline (vectorizer + SVD + model) to a directory.

    The resulting directory contains everything needed to reconstruct the model
    and run inference on raw text via `load_ann_run`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, out_dir / "vectorizer.pkl")
    joblib.dump(svd,        out_dir / "svd.pkl")

    # Strip torch.compile wrapper if present so the state_dict keys are clean
    state_dict = getattr(model, "_orig_mod", model).state_dict()

    torch.save(
        {
            "state_dict":  state_dict,
            "input_dim":   input_dim,
            "hidden_dims": hidden_dims,
            "num_classes": num_classes,
            "dropout":     dropout,
            "activation":  activation,
        },
        out_dir / "model.pt",
    )
    print(f"\nANN run saved to: {out_dir}")
    return out_dir


def load_ann_run(
    run_dir: str | Path,
    device: Optional[torch.device] = None,
) -> Tuple[SentimentANN, Any, Any]:
    """Load a saved ANN run.

    Returns
    -------
    (model, vectorizer, svd)
        Model is in eval mode on `device` (or CPU if not provided).
    """
    run_dir = Path(run_dir)
    if device is None:
        device = torch.device("cpu")

    vectorizer = joblib.load(run_dir / "vectorizer.pkl")
    svd        = joblib.load(run_dir / "svd.pkl")

    ckpt = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)
    activation_cls = _ACTIVATIONS.get(ckpt.get("activation", "relu"), nn.ReLU)

    model = SentimentANN(
        input_dim=ckpt["input_dim"],
        hidden_dims=ckpt["hidden_dims"],
        num_classes=ckpt["num_classes"],
        activation=activation_cls(),
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, vectorizer, svd


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Optional[Any] = None,
) -> Tuple[float, float]:
    """Run one full pass over `loader` in training mode.

    Returns (avg_loss, accuracy_%).

    Uses AMP automatically when CUDA is available.
    """
    device = next(model.parameters()).device
    cuda_available = torch.cuda.is_available()

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

def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Run a forward pass over `loader`, returning loss, accuracy, true labels, and predictions."""
    device = next(model.parameters()).device
    cuda_available = torch.cuda.is_available()

    running_loss = 0.0
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=cuda_available):
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    avg_loss = running_loss / len(y_true)
    accuracy = 100.0 * float((y_pred == y_true).sum()) / len(y_true)
    return avg_loss, accuracy, y_true, y_pred


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """Run one full pass over `loader` in evaluation mode.

    Returns (avg_loss, accuracy_%).
    """
    avg_loss, accuracy, _, _ = _collect_predictions(model, loader, criterion)
    return avg_loss, accuracy


def fit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    wandb_kwargs: Dict[str, Any],
    patience: Optional[int] = None,
    min_delta: float = 1e-4,
    scheduler=None,
    test_loader: Optional[DataLoader] = None,
) -> Dict[str, List[float]]:
    """Train `model` for up to `num_epochs`, validating after every epoch.

    Initialises and closes a wandb run for the duration of training.
    Saves the weights that achieved the best validation loss and restores
    them at the end.

    Parameters
    ----------
    model        : the network to train; modified in-place
    optimizer    : optimiser (e.g. Adam) already constructed for `model`
    criterion    : loss function (e.g. CrossEntropyLoss)
    train_loader : DataLoader for the training split
    val_loader   : DataLoader for the validation split
    num_epochs   : maximum number of epochs to train for
    wandb_kwargs : keyword arguments forwarded to wandb.init()
    patience     : number of consecutive epochs without improvement before
                   stopping early; None disables early stopping
    min_delta    : minimum absolute decrease in validation loss to count as
                   an improvement (filters out numerical noise)
    scheduler    : optional LR scheduler; stepped with val_loss each epoch
                   (ReduceLROnPlateau) or without args for all other types

    Returns
    -------
    history : dict with keys 'Training Loss', 'Validation Loss',
              'Training Accuracy', 'Validation Accuracy', each mapping to
              a list of per-epoch values
    """

    if torch.cuda.is_available():
        model = torch.compile(model)   # type: ignore[assignment]

    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None  # type: ignore[attr-defined]

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
        f"{'Epoch':<{epoch_col_w}} | "
        f"{'Train Loss':<{col_w}} | "
        f"{'Train Acc':<{col_w}} | "
        f"{'Val Loss':<{col_w}} | "
        f"{'Val Acc':<{col_w}}"
    )

    with wandb.init(**wandb_kwargs) as run:

        print(header)

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, scaler)
            val_loss,   val_acc   = validate(model, val_loader, criterion)

            history["Training Loss"].append(train_loss)
            history["Validation Loss"].append(val_loss)
            history["Training Accuracy"].append(train_acc)
            history["Validation Accuracy"].append(val_acc)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            early_stop = patience is not None and epochs_no_improve >= patience
            
            if epoch <= 5 or epoch % 5 == 0 or early_stop:  # Print first 5, every 5th, and the stopping epoch
                epoch_str = f"{epoch:>{w}}/{num_epochs:>{w}}"
                print(
                    f"{epoch_str:>{epoch_col_w}} | "
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

        # Log final test metrics if test_loader provided
        if test_loader is not None:
            test_loss, test_acc, y_true, y_pred = _collect_predictions(model, test_loader, criterion)
            macro_f1    = float(f1_score(y_true, y_pred, average='macro',    zero_division=0))
            weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            run.summary["test_loss"]        = test_loss
            run.summary["test_accuracy"]    = test_acc
            run.summary["test_macro_f1"]    = macro_f1
            run.summary["test_weighted_f1"] = weighted_f1

    return history

def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    label: str = "Test",
    class_names: Optional[List[str]] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """Evaluate `model` on `test_loader` and print a per-class classification report.

    Parameters
    ----------
    model       : trained model
    test_loader : DataLoader for the held-out test set
    criterion   : loss function
    label       : name printed in the summary line (e.g. the experiment name)
    class_names : optional list of human-readable class names for the report

    Returns
    -------
    (test_loss, test_acc_%, report_dict)
    """
    test_loss, test_acc, y_true, y_pred = _collect_predictions(model, test_loader, criterion)

    print(f"Classification Report: {label}\n")

    print(classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=3,
        zero_division=0,
    ))

    report_dict = cast(Dict[str, Any], classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=3,
        zero_division=0,
        output_dict=True,
    ))
    return test_loss, test_acc, report_dict