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

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Dataset
from torchmetrics import ConfusionMatrix


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
        texts: Sequence[str],
        labels: Sequence[int],
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
    device: torch.device,
    scheduler: Optional[Any] = None,
) -> Tuple[float, float]:
    """Run one training epoch for a BERT-based classifier.

    Returns
    -------
    avg_loss, accuracy_%
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        batch = _batch_to_device(batch, device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        logits = outputs.logits
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch["labels"].size(0)
        correct += (logits.argmax(dim=1) == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate_bert(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one evaluation epoch for a BERT-based classifier.

    Returns
    -------
    avg_loss, accuracy_%
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            logits = outputs.logits
            loss = criterion(logits, batch["labels"])

            running_loss += loss.item() * batch["labels"].size(0)
            correct += (logits.argmax(dim=1) == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def fit_bert(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    wandb_kwargs: Dict[str, Any],
    scheduler: Optional[Any] = None,
    log: bool = True,
) -> Dict[str, List[float]]:
    """Train a BERT-based model with validation after every epoch.

    The best checkpoint is selected by validation loss and restored at the end.
    """
    if not log:
        wandb_kwargs = {**wandb_kwargs, "mode": "disabled"}

    with wandb.init(**wandb_kwargs):
        best_val_loss = float("inf")
        best_state = None

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train_bert(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scheduler=scheduler,
            )

            val_loss, val_acc = validate_bert(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(
                f"Epoch {epoch:>{len(str(num_epochs))}}/{num_epochs} | "
                f"train loss {train_loss:.4f}, train acc {train_acc:.2f}% | "
                f"val loss {val_loss:.4f}, val acc {val_acc:.2f}%"
            )

            wandb.log({
                "Training Loss": train_loss,
                "Training Accuracy": train_acc,
                "Validation Loss": val_loss,
                "Validation Accuracy": val_acc,
            })

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best weights (val loss {best_val_loss:.4f})")

    return history


def evaluate_bert(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label: str = "Test",
) -> Tuple[float, float]:
    """Evaluate a BERT-based model on the held-out test loader."""
    test_loss, test_acc = validate_bert(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )
    print(f"[{label}] Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")
    return test_loss, test_acc


def plot_confusion_matrix_bert(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Collect predictions from a BERT loader and plot a confusion matrix."""
    metric = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            preds = outputs.logits.argmax(dim=1)
            metric.update(preds, batch["labels"])

    cm = metric.compute()

    labels_ = class_names if class_names else [str(i) for i in range(num_classes)]
    fig, ax = plt.subplots(figsize=(max(4, num_classes), max(4, num_classes)))
    im = ax.imshow(cm.cpu().numpy(), cmap="Blues")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(labels_, rotation=45, ha="right")
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(labels_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, pad=10, y=1.02)

    cm_max = cm.max().item()
    for i in range(num_classes):
        for j in range(num_classes):
            count = cm[i, j].item()
            color = "white" if count > cm_max * 0.6 else "black"
            ax.text(j, i, str(int(count)), ha="center", va="center", color=color)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
