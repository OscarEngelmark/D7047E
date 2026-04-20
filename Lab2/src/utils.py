import sys
import platform
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt


def device_check() -> torch.device:
    """Detect the best available device and print a summary of the environment.

    Prefers CUDA, then MPS (Apple Silicon), then falls back to CPU.
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
        printout = "Using cpu"

    print(printout)
    return device


def show_generated_images(
    generator: torch.nn.Module,
    latent_dim: int,
    device: torch.device,
    num_images: int = 16,
) -> None:
    """Sample from generator and display a grid in the notebook."""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, device=device)
        fake_images = generator(z).view(-1, 28, 28).cpu().numpy()

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake_images[i], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    generator.train()


def make_generated_figure(
    generator: torch.nn.Module,
    latent_dim: int,
    device: torch.device,
    num_images: int = 16,
) -> plt.Figure:
    """Sample from generator and return a matplotlib figure for W&B logging."""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, device=device)
        fake_images = generator(z).view(-1, 28, 28).cpu().numpy()

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake_images[i], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    generator.train()
    return fig


def save_generated_grid(
    generator: torch.nn.Module,
    latent_dim: int,
    save_path: str | Path,
    device: torch.device,
    num_images: int = 16,
) -> None:
    """Sample from generator and save a grid image to disk."""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, device=device)
        fake_images = generator(z).view(-1, 28, 28).cpu().numpy()

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake_images[i], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    generator.train()


def build_model_name(config: dict, task_name: str = "task1", file_ext: str = "pt") -> str:
    """Build a readable checkpoint filename from a GAN training config."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{task_name}"
        f"_model-{config['model'].replace(' ', '_')}"
        f"_data-{config['dataset']}"
        f"_ep-{config['epochs']}"
        f"_bs-{config['batch_size']}"
        f"_glr-{config['g_lr']}"
        f"_dlr-{config['d_lr']}"
        f"_gh-{config['g_hidden_dim']}"
        f"_dh-{config['d_hidden_dim']}"
        f"_seed-{config['seed']}"
        f"_{timestamp}.{file_ext}"
    )
