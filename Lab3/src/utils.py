import platform
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from PIL import Image
from tqdm.auto import tqdm


def device_check() -> torch.device:
    """Detect compute device and print environment info."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("Python version:", sys.version)
    print("OS:", platform.system(), platform.release())
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print(
            "CUDA version used by PyTorch:", torch.version.cuda
        )
    return device


def load_captions_file(captions_path: Path) -> pd.DataFrame:
    """Load a captions CSV into a DataFrame."""
    df = pd.read_csv(captions_path)

    if "image" not in df.columns or "caption" not in df.columns:
        df = pd.read_csv(
            captions_path, names=["image", "caption"]
        )

    df["image"] = df["image"].astype(str).apply(
        lambda x: Path(x).name
    )
    df["caption"] = df["caption"].astype(str)

    return df


def clean_caption(text: str) -> str:
    """Lowercase, strip punctuation, and add start/end tokens."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [word for word in text.split() if len(word) > 1]
    text = " ".join(tokens)
    return f"<start> {text} <end>"


def show_image_with_captions(
    image_name: str,
    captions_mapping: Dict[str, List[str]],
    image_dir: Path,
) -> None:
    """Display an image and its reference captions."""
    image_path = image_dir / image_name
    image = Image.open(image_path).convert("RGB")

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    print("Reference captions:")
    for caption in captions_mapping[image_name]:
        print("-", caption)


class AverageMeter:
    """Track and update a running average."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0


def accuracy_topk(
    scores: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
) -> float:
    """Compute top-k accuracy."""
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = scores.topk(k, dim=1)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        correct_total = correct.reshape(-1).float().sum().item()
    return correct_total * (100.0 / batch_size)


def save_checkpoint(
    run_dir: Path,
    epoch: int,
    encoder: nn.Module,
    decoder: nn.Module,
    decoder_optimizer: torch.optim.Optimizer,
    encoder_optimizer: Optional[torch.optim.Optimizer],
    best_val_loss: float,
    history: Dict,
    vocab: object,
    model_config: Dict,
    is_best: bool = False,
) -> Path:
    """Save a training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "decoder_optimizer_state_dict": (
            decoder_optimizer.state_dict()
        ),
        "encoder_optimizer_state_dict": (
            encoder_optimizer.state_dict()
            if encoder_optimizer is not None
            else None
        ),
        "best_val_loss": best_val_loss,
        "history": history,
        "vocab": vocab,
        "model_config": model_config,
    }

    latest_path = run_dir / "checkpoint_latest.pth"
    torch.save(checkpoint, latest_path)

    if is_best:
        best_path = run_dir / "checkpoint_best.pth"
        torch.save(checkpoint, best_path)

    return latest_path


def generate_caption_greedy(
    encoder: nn.Module,
    decoder: nn.Module,
    image: torch.Tensor,
    vocab: object,
    device: torch.device,
    max_length: int = 30,
) -> str:
    """Generate a caption for one image using greedy decoding."""
    encoder.eval()
    decoder.eval()

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_out = encoder(image)

        h, c = decoder.init_hidden_state(encoder_out)

        word_id = torch.tensor(
            [vocab.word2idx[vocab.start_token]],
            dtype=torch.long,
            device=device,
        )

        generated_ids = []

        for _ in range(max_length):
            embedding = decoder.embedding(word_id)

            context, alpha = decoder.attention(encoder_out, h)

            gate = decoder.sigmoid(decoder.f_beta(h))
            context = gate * context

            h, c = decoder.decode_step(
                torch.cat([embedding, context], dim=1),
                (h, c),
            )

            scores = decoder.fc(h)
            predicted_id = scores.argmax(dim=1)
            predicted_id_int = int(predicted_id.item())

            if predicted_id_int == vocab.word2idx[vocab.end_token]:
                break

            generated_ids.append(predicted_id_int)
            word_id = predicted_id

    return vocab.decode_ids(generated_ids, remove_special=True)


def show_prediction_pytorch(
    image_name: str,
    encoder: nn.Module,
    decoder: nn.Module,
    vocab: object,
    device: torch.device,
    image_to_captions: Dict[str, List[str]],
    image_dir: Path,
    dataset_transform: Callable,
) -> None:
    """Show one image with generated and reference captions."""
    image_path = image_dir / image_name

    raw_image = Image.open(image_path).convert("RGB")
    model_image = dataset_transform(raw_image)

    generated_caption = generate_caption_greedy(
        encoder=encoder,
        decoder=decoder,
        image=model_image,
        vocab=vocab,
        device=device,
        max_length=30,
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(raw_image)
    plt.axis("off")
    plt.title("Generated: " + generated_caption)
    plt.show()

    print("Generated caption:")
    print(generated_caption)

    print("\nReference captions:")
    for caption in image_to_captions[image_name]:
        clean_ref = (
            caption
            .replace("<start>", "")
            .replace("<end>", "")
            .strip()
        )
        print("-", clean_ref)


def generate_caption_beam_search(
    encoder: nn.Module,
    decoder: nn.Module,
    image: torch.Tensor,
    vocab: object,
    device: torch.device,
    beam_size: int = 3,
    max_length: int = 30,
) -> str:
    """Generate a caption for one image using beam search."""
    encoder.eval()
    decoder.eval()

    start_token = vocab.word2idx[vocab.start_token]
    end_token = vocab.word2idx[vocab.end_token]

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_out = encoder(image)

        # Repeat encoder output for each beam.
        encoder_out = encoder_out.expand(
            beam_size,
            encoder_out.size(1),
            encoder_out.size(2),
        )

        h, c = decoder.init_hidden_state(encoder_out)

        sequences = torch.full(
            size=(beam_size, 1),
            fill_value=start_token,
            dtype=torch.long,
            device=device,
        )

        top_k_scores = torch.zeros(beam_size, 1, device=device)

        complete_sequences: List[List[int]] = []
        complete_scores: List[float] = []

        word_ids = sequences[:, -1]

        for step in range(max_length):
            embeddings = decoder.embedding(word_ids)

            context, alpha = decoder.attention(encoder_out, h)

            gate = decoder.sigmoid(decoder.f_beta(h))
            context = gate * context

            h, c = decoder.decode_step(
                torch.cat([embeddings, context], dim=1),
                (h, c),
            )

            scores = decoder.fc(h)
            log_probs = torch.log_softmax(scores, dim=1)

            scores = top_k_scores.expand_as(log_probs) + log_probs

            if step == 0:
                top_k_scores, top_k_words = scores[0].topk(
                    beam_size, dim=0
                )
            else:
                top_k_scores, top_k_words = (
                    scores.view(-1).topk(beam_size, dim=0)
                )

            previous_beam_indices = top_k_words // len(vocab)
            next_word_ids = top_k_words % len(vocab)

            sequences = torch.cat(
                [
                    sequences[previous_beam_indices],
                    next_word_ids.unsqueeze(1),
                ],
                dim=1,
            )

            h = h[previous_beam_indices]
            c = c[previous_beam_indices]
            encoder_out = encoder_out[previous_beam_indices]

            incomplete_indices = [
                idx
                for idx, word_id in enumerate(next_word_ids)
                if int(word_id.item()) != end_token
            ]

            complete_indices = [
                idx
                for idx, word_id in enumerate(next_word_ids)
                if int(word_id.item()) == end_token
            ]

            if len(complete_indices) > 0:
                for idx in complete_indices:
                    complete_sequences.append(
                        sequences[idx].tolist()
                    )
                    complete_scores.append(
                        float(top_k_scores[idx].item())
                    )

            beam_size = len(incomplete_indices)

            if beam_size == 0:
                break

            sequences = sequences[incomplete_indices]
            h = h[incomplete_indices]
            c = c[incomplete_indices]
            encoder_out = encoder_out[incomplete_indices]
            top_k_scores = (
                top_k_scores[incomplete_indices].unsqueeze(1)
            )
            word_ids = next_word_ids[incomplete_indices]

        if len(complete_sequences) > 0:
            best_index = int(np.argmax(complete_scores))
            best_sequence = complete_sequences[best_index]
        else:
            best_sequence = sequences[0].tolist()

    words = []

    for token_id in best_sequence:
        word = vocab.idx2word.get(int(token_id), vocab.unk_token)

        if word in {
            vocab.start_token,
            vocab.end_token,
            vocab.pad_token,
        }:
            continue

        words.append(word)

    return " ".join(words)


def clean_reference_caption(caption: str) -> str:
    """Remove special tokens from a reference caption."""
    return (
        caption
        .replace("<start>", "")
        .replace("<end>", "")
        .strip()
    )


def evaluate_bleu(
    encoder: nn.Module,
    decoder: nn.Module,
    image_names: List[str],
    image_to_captions: Dict[str, List[str]],
    vocab: object,
    device: torch.device,
    image_dir: Path,
    eval_transform: Callable,
    decoding_method: str = "beam",
    beam_size: int = 3,
    max_length: int = 30,
) -> Dict[str, float]:
    """Evaluate generated captions using BLEU-1 through BLEU-4."""
    smooth = SmoothingFunction().method1
    actual_captions: List[List[List[str]]] = []
    predicted_captions: List[List[str]] = []

    desc = f"Evaluating BLEU with {decoding_method} decoding"

    for image_name in tqdm(image_names, desc=desc):
        image_path = image_dir / image_name
        raw_image = Image.open(image_path).convert("RGB")
        model_image = eval_transform(raw_image)

        references = [
            clean_reference_caption(caption).split()
            for caption in image_to_captions[image_name]
        ]

        if decoding_method == "greedy":
            prediction = generate_caption_greedy(
                encoder=encoder,
                decoder=decoder,
                image=model_image,
                vocab=vocab,
                device=device,
                max_length=max_length,
            )
        elif decoding_method == "beam":
            prediction = generate_caption_beam_search(
                encoder=encoder,
                decoder=decoder,
                image=model_image,
                vocab=vocab,
                device=device,
                beam_size=beam_size,
                max_length=max_length,
            )
        else:
            raise ValueError(
                "decoding_method must be 'greedy' or 'beam'."
            )

        actual_captions.append(references)
        predicted_captions.append(prediction.split())

    bleu_1 = corpus_bleu(
        actual_captions,
        predicted_captions,
        weights=(1.0, 0, 0, 0),
        smoothing_function=smooth,
    )
    bleu_2 = corpus_bleu(
        actual_captions,
        predicted_captions,
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smooth,
    )
    bleu_3 = corpus_bleu(
        actual_captions,
        predicted_captions,
        weights=(1 / 3, 1 / 3, 1 / 3, 0),
        smoothing_function=smooth,
    )
    bleu_4 = corpus_bleu(
        actual_captions,
        predicted_captions,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth,
    )

    return {
        "BLEU-1": bleu_1,
        "BLEU-2": bleu_2,
        "BLEU-3": bleu_3,
        "BLEU-4": bleu_4,
    }
