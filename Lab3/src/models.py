import random
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights, resnet50


class Vocabulary:
    """
    A simple word-level vocabulary for image captioning.
    """

    def __init__(self, min_freq: int = 5) -> None:
        self.min_freq = min_freq

        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.start_token = "<start>"
        self.end_token = "<end>"

        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

    def build(self, captions: List[str]) -> None:
        """Build the vocabulary from a list of captions."""
        counter: Counter = Counter()

        for caption in captions:
            counter.update(caption.split())

        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.start_token,
            self.end_token,
        ]

        for token in special_tokens:
            self._add_word(token)

        for word, count in counter.items():
            if count >= self.min_freq and word not in self.word2idx:
                self._add_word(word)

    def _add_word(self, word: str) -> None:
        """Add one word to the vocabulary."""
        idx = len(self.word2idx)
        self.word2idx[word] = idx
        self.idx2word[idx] = word

    def numericalize(self, caption: str) -> List[int]:
        """Convert a caption string into a list of token IDs."""
        return [
            self.word2idx.get(word, self.word2idx[self.unk_token])
            for word in caption.split()
        ]

    def decode_ids(
        self,
        token_ids: List[int],
        remove_special: bool = True,
    ) -> str:
        """Convert token IDs back into a caption string."""
        words = []

        for token_id in token_ids:
            word = self.idx2word.get(int(token_id), self.unk_token)

            if remove_special and word in {
                self.pad_token,
                self.start_token,
                self.end_token,
            }:
                continue

            words.append(word)

        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)


class Flickr8kCaptionDataset(Dataset):
    """
    Flickr8k dataset for image captioning.
    """

    def __init__(
        self,
        image_names: List[str],
        captions_mapping: Dict[str, List[str]],
        image_dir: Path,
        vocab: Vocabulary,
        transform: Optional[Callable[..., torch.Tensor]] = None,
        random_caption: bool = True,
    ) -> None:
        self.image_names = list(image_names)
        self.captions_mapping = captions_mapping
        self.image_dir = Path(image_dir)
        self.vocab = vocab
        self.transform = transform
        self.random_caption = random_caption

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        image_name = self.image_names[index]
        image_path = self.image_dir / image_name

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        tensor_image = cast(torch.Tensor, image)
        captions = self.captions_mapping[image_name]

        if self.random_caption:
            caption = random.choice(captions)
        else:
            caption = captions[0]

        caption_ids = torch.tensor(
            self.vocab.numericalize(caption),
            dtype=torch.long,
        )

        return tensor_image, caption_ids, image_name


def make_collate_fn(pad_idx: int) -> Callable:
    """Return a collate function for variable-length captions."""

    def caption_collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor, str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        Pad captions to the same length within a batch.

        Returns:
            images: Tensor [batch_size, 3, 224, 224]
            captions: LongTensor [batch_size, max_caption_length]
            lengths: LongTensor of actual caption lengths
            image_names: list of image filenames
        """
        images, captions, image_names = zip(*batch)

        images = torch.stack(images, dim=0)

        lengths = torch.tensor(
            [len(caption) for caption in captions],
            dtype=torch.long,
        )

        max_len = int(lengths.max().item())

        padded_captions = torch.full(
            size=(len(captions), max_len),
            fill_value=pad_idx,
            dtype=torch.long,
        )

        for i, caption in enumerate(captions):
            padded_captions[i, : len(caption)] = caption

        return images, padded_captions, lengths, list(image_names)

    return caption_collate_fn


class EncoderCNN(nn.Module):
    """
    CNN encoder based on pretrained ResNet50.

    The final average pooling and classification layers are removed.
    The output is a spatial feature map with shape:
        [batch_size, num_pixels, encoder_dim]
    """

    def __init__(
        self,
        encoded_image_size: int = 14,
        fine_tune: bool = False,
    ) -> None:
        super().__init__()

        self.encoded_image_size = encoded_image_size

        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)

        # Remove avgpool and fc layers.
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize feature map to a fixed spatial size.
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

        self.fine_tune(fine_tune)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: Tensor [batch_size, 3, 224, 224]

        Returns:
            features: Tensor [batch_size, num_pixels, encoder_dim]
        """
        features = self.resnet(images)
        features = self.adaptive_pool(features)

        # [batch_size, encoder_dim, H, W] -> [batch_size, H, W, encoder_dim]
        features = features.permute(0, 2, 3, 1)

        batch_size = features.size(0)
        encoder_dim = features.size(-1)

        # Flatten spatial dimensions: [batch_size, H*W, encoder_dim]
        features = features.view(batch_size, -1, encoder_dim)

        return features

    def fine_tune(self, fine_tune: bool = True) -> None:
        """Enable or disable fine-tuning of ResNet layers."""
        for param in self.resnet.parameters():
            param.requires_grad = False

        if fine_tune:
            # Fine-tune only later ResNet blocks.
            for child in list(self.resnet.children())[5:]:
                for param in child.parameters():
                    param.requires_grad = True


class Attention(nn.Module):
    """
    Additive attention mechanism.

    Computes attention weights over image regions using the
    current decoder hidden state.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        attention_dim: int,
    ) -> None:
        super().__init__()

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_out: Tensor [batch_size, num_pixels, encoder_dim]
            decoder_hidden: Tensor [batch_size, decoder_dim]

        Returns:
            context: Tensor [batch_size, encoder_dim]
            alpha: Tensor [batch_size, num_pixels]
        """
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)

        attention_scores = self.full_att(
            self.relu(att1 + att2.unsqueeze(1))
        ).squeeze(2)

        alpha = self.softmax(attention_scores)

        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return context, alpha


class DecoderRNNWithAttention(nn.Module):
    """
    LSTM decoder with attention.

    At each decoding step:
    - compute attention over image features
    - combine attended visual context with word embedding
    - update LSTM hidden state
    - predict the next word
    """

    def __init__(
        self,
        attention_dim: int,
        embed_dim: int,
        decoder_dim: int,
        vocab_size: int,
        encoder_dim: int = 2048,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.attention = Attention(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim,
        )

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.decode_step = nn.LSTMCell(
            input_size=embed_dim + encoder_dim,
            hidden_size=decoder_dim,
        )

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(decoder_dim, vocab_size)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize embedding and classification layers."""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(
        self, encoder_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM states from the mean image feature."""
        mean_encoder_out = encoder_out.mean(dim=1)

        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)

        return h, c

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoded_captions: torch.Tensor,
        caption_lengths: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[int],
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Forward pass for training.

        Args:
            encoder_out: Tensor [batch_size, num_pixels, encoder_dim]
            encoded_captions: Tensor [batch_size, max_caption_length]
            caption_lengths: Tensor [batch_size]

        Returns:
            predictions: Tensor [batch_size, max_decode_len, vocab_size]
            encoded_captions: sorted encoded captions
            decode_lengths: list of decoding lengths
            alphas: attention weights
            sort_ind: sorting indices
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing caption lengths for efficient decoding
        caption_lengths, sort_ind = caption_lengths.sort(
            dim=0, descending=True
        )
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embed captions: [batch_size, max_caption_length, embed_dim]
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state with mean image feature
        h, c = self.init_hidden_state(encoder_out)

        # We never feed <end> as decoder input
        decode_lengths = (caption_lengths - 1).tolist()
        max_decode_length = max(decode_lengths)

        predictions = torch.zeros(
            batch_size,
            max_decode_length,
            self.vocab_size,
            device=encoder_out.device,
        )

        alphas = torch.zeros(
            batch_size,
            max_decode_length,
            num_pixels,
            device=encoder_out.device,
        )

        for t in range(max_decode_length):
            batch_size_t = sum(length > t for length in decode_lengths)

            context, alpha = self.attention(
                encoder_out[:batch_size_t],
                h[:batch_size_t],
            )

            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            context = gate * context

            h_t, c_t = self.decode_step(
                torch.cat(
                    [embeddings[:batch_size_t, t, :], context],
                    dim=1,
                ),
                (h[:batch_size_t], c[:batch_size_t]),
            )

            preds = self.fc(self.dropout(h_t))

            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

            h = h.clone()
            c = c.clone()
            h[:batch_size_t] = h_t
            c[:batch_size_t] = c_t

        return (
            predictions,
            encoded_captions,
            decode_lengths,
            alphas,
            sort_ind,
        )
