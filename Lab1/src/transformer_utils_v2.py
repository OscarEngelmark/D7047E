"""transformer_utils_v2.py — Trainer-based helpers for sentiment classification.

Drop-in replacement for transformer_utils.py that delegates training to
Hugging Face Trainer / TrainingArguments, keeping only the bits the stock
Trainer doesn't cover: class-weighted loss, per-class classification report,
and confusion-matrix plotting.

Exports
-------
BertSentimentDataset     : torch Dataset that tokenizes text for BERT
build_tf_datasets        : build train/val/test datasets from DataFrames
compute_metrics_tf       : accuracy + macro/weighted F1 for Trainer
WeightedTrainer          : Trainer subclass with class-weighted CE loss
evaluate_tf              : per-class classification report from a trained Trainer
plot_confusion_matrix_tf : confusion matrix plot from a trained Trainer
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset
from torchmetrics import ConfusionMatrix
from transformers import Trainer

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline as _backend_inline

# Render inline plots as SVG so all edges stay crisp regardless of screen DPI.
_backend_inline.set_matplotlib_formats("svg")


class BertSentimentDataset(Dataset):
    """Torch Dataset that tokenizes text for BERT-family classifiers.

    Returns dicts with ``input_ids``, ``attention_mask``, and ``labels``,
    which matches the signature expected by Hugging Face Trainer.
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
        enc = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def build_tf_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: Any,
    text_col: str = "Sentence",
    label_col: str = "Class",
    max_length: int = 128,
) -> Dict[str, BertSentimentDataset]:
    """Build train / val / test datasets for Trainer from DataFrames."""
    return {
        split: BertSentimentDataset(
            texts=df[text_col],
            labels=df[label_col],
            tokenizer=tokenizer,
            max_length=max_length,
        )
        for split, df in (("train", train_df), ("val", val_df), ("test", test_df))
    }


def compute_metrics_tf(eval_pred: Any) -> Dict[str, float]:
    """Compute accuracy, macro-F1, and weighted-F1 from Trainer eval output."""
    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(axis=-1)
    labels = np.asarray(labels)
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


class WeightedTrainer(Trainer):
    """Trainer variant that applies class weights to the cross-entropy loss.

    Used for imbalanced datasets like the Video Game 5-star split. Honours
    the ``label_smoothing_factor`` set on TrainingArguments so behaviour
    matches the stock Trainer for everything except the weighting.
    """

    def __init__(
        self,
        *args: Any,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Any:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)

        loss = F.cross_entropy(
            logits, labels,
            weight=weight,
            label_smoothing=self.args.label_smoothing_factor,
        )
        return (loss, outputs) if return_outputs else loss


def evaluate_tf(
    trainer: Trainer,
    test_dataset: Dataset,
    label: str = "Test",
    class_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Run Trainer.predict on the test set and print a classification report.

    Returns ``(report_dict, y_true, y_pred)`` so the predictions can be
    reused (e.g. for plotting) without a second forward pass.
    """
    pred = trainer.predict(test_dataset, metric_key_prefix="test")
    y_true = np.asarray(pred.label_ids)
    y_pred = np.asarray(pred.predictions).argmax(axis=-1)

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

    if pred.metrics:
        summary = {k: v for k, v in pred.metrics.items() if isinstance(v, (int, float))}
        if summary:
            print("Trainer metrics:")
            for k, v in summary.items():
                print(f"  {k}: {v:.4f}")

    return report_dict, y_true, y_pred


def plot_confusion_matrix_tf(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    """Plot a confusion matrix from precomputed predictions."""
    y_true_t = torch.as_tensor(np.asarray(y_true), dtype=torch.long)
    y_pred_t = torch.as_tensor(np.asarray(y_pred), dtype=torch.long)

    task = cast(
        Literal["binary", "multiclass", "multilabel"],
        "binary" if num_classes == 2 else "multiclass",
    )
    metric = ConfusionMatrix(task=task, num_classes=num_classes)
    metric.update(y_pred_t, y_true_t)
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

    plot_dir = Path("../plot")
    plot_dir.mkdir(parents=True, exist_ok=True)

    safe_title = title.lower().replace(" ", "_").replace("/", "_")
    suffix = "_normalized" if normalize else ""
    save_path = plot_dir / f"{safe_title}{suffix}.png"

    plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    plt.show()
