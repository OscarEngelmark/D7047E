"""utils_BERT.py — BERT-specific training helpers for sentiment classification.

Exports
-------
BertSentimentDataset   : Dataset wrapper that tokenizes text for BERT
build_bert_loaders     : create train / val / test DataLoaders from DataFrames
train_bert             : one training epoch
validate_bert          : one evaluation epoch
fit_bert               : full training loop with wandb logging and best-checkpoint restore
evaluate_bert          : evaluate a trained model on the test loader and print a summary
plot_confusion_matrix_bert : plot confusion matrix for a BERT classifier
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast

import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset
from torchmetrics import ConfusionMatrix
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline as _backend_inline

# Render inline plots as SVG so all edges stay crisp regardless of screen DPI.
_backend_inline.set_matplotlib_formats("svg")


class BertSentimentDataset(Dataset):
    """Dataset wrapper for BERT-based sentiment classification.

    Parameters
    ----------
    texts   : sequence of raw text strings
    labels  : sequence of integer class labels
    tokenizer: Hugging Face tokenizer
    max_length: maximum tokenized sequence length
    """

    def __init__(
        self,
        texts: Sequence[str] | pd.Series,
        labels: Sequence[int] | pd.Series,
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def build_bert_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: Any,
    text_col: str = "Text",
    label_col: str = "Class",
    max_length: int = 128,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for BERT from train / val / test DataFrames."""

    train_dataset = BertSentimentDataset(
        texts=train_df[text_col],
        labels=train_df[label_col],
        tokenizer=tokenizer,
        max_length=max_length,
    )

    val_dataset = BertSentimentDataset(
        texts=val_df[text_col],
        labels=val_df[label_col],
        tokenizer=tokenizer,
        max_length=max_length,
    )

    test_dataset = BertSentimentDataset(
        texts=test_df[text_col],
        labels=test_df[label_col],
        tokenizer=tokenizer,
        max_length=max_length,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def _batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move a BERT batch to the target device."""
    return {key: value.to(device) for key, value in batch.items()}


def train_bert(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> Tuple[float, float]:
    """Run one training epoch for a BERT-based classifier.

    Returns
    -------
    avg_loss, accuracy_%
    """
    device = next(model.parameters()).device
    cuda_available = torch.cuda.is_available()

    running_loss = 0.0
    correct = 0
    total = 0

    model.train()

    for batch in loader:
        batch = _batch_to_device(batch, device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=cuda_available):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            loss = criterion(logits, batch["labels"])

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch["labels"].size(0)
        correct += (logits.argmax(dim=1) == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

    return running_loss / total, 100.0 * correct / total


def validate_bert(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """Run one evaluation epoch for a BERT-based classifier.

    Returns
    -------
    avg_loss, accuracy_%
    """
    device = next(model.parameters()).device
    cuda_available = torch.cuda.is_available()

    running_loss = 0.0
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)

            with torch.autocast(device_type=device.type, enabled=cuda_available):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs.logits
                loss = criterion(logits, batch["labels"])

            running_loss += loss.item() * batch["labels"].size(0)
            correct += (logits.argmax(dim=1) == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    return running_loss / total, 100.0 * correct / total


def fit_bert(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    wandb_kwargs: Dict[str, Any],
    scheduler: Optional[Any] = None,
    log: bool = True,
    patience: Optional[int] = None,
    min_delta: float = 1e-4,
) -> Dict[str, List[float]]:
    """Train a BERT-based model with validation after every epoch.

    The best checkpoint is selected by validation loss and restored at the end.
    The scheduler (if provided) is stepped per batch inside train_bert, which
    is appropriate for warmup-style BERT schedulers.
    """
    if not log:
        wandb_kwargs = {**wandb_kwargs, "mode": "disabled"}

    if torch.cuda.is_available():
        model = torch.compile(model)  # type: ignore[assignment]

    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None  # type: ignore[attr-defined]

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0
    history: Dict[str, List[float]] = {
        "Training Loss": [],
        "Validation Loss": [],
        "Training Accuracy": [],
        "Validation Accuracy": [],
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
            train_loss, train_acc = train_bert(model, train_loader, optimizer, criterion, scheduler, scaler)
            val_loss,   val_acc   = validate_bert(model, val_loader, criterion)

            history["Training Loss"].append(train_loss)
            history["Validation Loss"].append(val_loss)
            history["Training Accuracy"].append(train_acc)
            history["Validation Accuracy"].append(val_acc)

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            early_stop = patience is not None and epochs_no_improve >= patience

            if epoch <= 5 or epoch % 5 == 0 or early_stop:
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

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best weights (val loss {best_val_loss:.4f})")

    return history


def evaluate_bert(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    label: str = "Test",
) -> Tuple[float, float]:
    """Evaluate a BERT-based model on the held-out test loader."""
    test_loss, test_acc = validate_bert(model, test_loader, criterion)
    print(f"[{label}] Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")
    return test_loss, test_acc


def plot_confusion_matrix_bert(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    """Collect predictions from a BERT loader and plot a confusion matrix."""

    device = next(model.parameters()).device
    task = cast(Literal["binary", "multiclass", "multilabel"],
                "binary" if num_classes == 2 else "multiclass")
    metric = ConfusionMatrix(task=task, num_classes=num_classes).to(device)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            preds = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits.argmax(dim=1)
            metric.update(preds, batch["labels"])

    cm = metric.compute()

    if normalize:
        row_sums = cm.sum(dim=1, keepdim=True).clamp(min=1)
        cm_display = (cm.float() / row_sums).cpu().numpy()
    else:
        cm_display = cm.cpu().numpy()

    labels_ = class_names if class_names else [str(i) for i in range(num_classes)]
    fig, ax = plt.subplots(figsize=(max(4, num_classes), max(4, num_classes)))
    im = ax.imshow(cm_display, cmap="Blues", vmin=0, vmax=(1.0 if normalize else None))

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(labels_, rotation=45, ha="right")
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(labels_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, pad=10, y=1.02)

    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_display[i, j]
            text = f"{val:.2f}" if normalize else str(int(cm[i, j].item()))
            color = "white" if val > (0.6 if normalize else cm_display.max() * 0.6) else "black"
            ax.text(j, i, text, ha="center", va="center", color=color)

    plt.tight_layout(rect=(0, 0, 1, 0.97))

    plot_dir = Path("plot")
    plot_dir.mkdir(parents=True, exist_ok=True)

    safe_title = title.lower().replace(" ", "_").replace("/", "_")
    suffix = "_normalized" if normalize else ""
    save_path = plot_dir / f"{safe_title}{suffix}.png"

    plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    plt.show()