import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import wandb

from tqdm.auto import tqdm

from utils import make_generated_figure, make_cgan_all_digits_figure


def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, x_dim)
        self.apply(xavier_init)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        out = torch.sigmoid(self.fc2(h))
        return out


class Discriminator(nn.Module):
    def __init__(self, x_dim, h_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)
        self.apply(xavier_init)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)


class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(z_dim + num_classes, h_dim)
        self.fc2 = nn.Linear(h_dim, x_dim)
        self.apply(xavier_init)

    def forward(self, z, labels):
        y = F.one_hot(labels, self.num_classes).float()
        h = F.relu(self.fc1(torch.cat([z, y], dim=1)))
        return torch.sigmoid(self.fc2(h))


class ConditionalDiscriminator(nn.Module):
    def __init__(self, x_dim, h_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(x_dim + num_classes, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)
        self.apply(xavier_init)

    def forward(self, x, labels):
        y = F.one_hot(labels, self.num_classes).float()
        h = F.relu(self.fc1(torch.cat([x, y], dim=1)))
        return self.fc2(h)


def train_GAN_epoch(
    G: nn.Module,
    D: nn.Module,
    train_loader,
    g_optimizer,
    d_optimizer,
    latent_dim: int,
    image_dim: int,
    device: torch.device,
    loss_type: str = "bce",
    scaler=None,
) -> tuple[float, float, float, float]:
    """Run one training epoch for a vanilla GAN.

    Returns avg_d_loss, avg_g_loss, avg_d_real_loss, avg_d_fake_loss.
    loss_type: "bce" uses BCEWithLogitsLoss internally; "logistic" uses softplus-based loss.
    """
    G.train()
    D.train()

    criterion = nn.BCEWithLogitsLoss()

    amp_enabled = scaler is not None
    d_loss_list, g_loss_list, d_real_loss_list, d_fake_loss_list = [], [], [], []

    for real_images, _ in train_loader:
        real_images = real_images.view(-1, image_dim).to(device, non_blocking=True)
        batch_size = real_images.size(0)

        # Train Discriminator
        z = torch.randn(batch_size, latent_dim, device=device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            d_real = D(real_images)
            d_fake = D(G(z).detach())
            if loss_type == "logistic":
                d_real_loss = F.softplus(-d_real).mean()
                d_fake_loss = F.softplus(d_fake).mean()
            else:
                real_targets = torch.ones(batch_size, 1, device=device)
                fake_targets = torch.zeros(batch_size, 1, device=device)
                d_real_loss = criterion(d_real, real_targets)
                d_fake_loss = criterion(d_fake, fake_targets)
            d_loss = d_real_loss + d_fake_loss
        d_optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(d_loss).backward()
            scaler.step(d_optimizer)
            scaler.update()
        else:
            d_loss.backward()
            d_optimizer.step()

        # Train Generator
        z = torch.randn(batch_size, latent_dim, device=device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            fake_logits = D(G(z))
            if loss_type == "logistic":
                g_loss = F.softplus(-fake_logits).mean()
            else:
                real_targets = torch.ones(batch_size, 1, device=device)
                g_loss = criterion(fake_logits, real_targets)
        g_optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(g_loss).backward()
            scaler.step(g_optimizer)
            scaler.update()
        else:
            g_loss.backward()
            g_optimizer.step()

        d_loss_list.append(d_loss.item())
        g_loss_list.append(g_loss.item())
        d_real_loss_list.append(d_real_loss.item())
        d_fake_loss_list.append(d_fake_loss.item())

    return (
        float(np.mean(d_loss_list)),
        float(np.mean(g_loss_list)),
        float(np.mean(d_real_loss_list)),
        float(np.mean(d_fake_loss_list)),
    )


def train_cGAN_epoch(
    G: nn.Module,
    D: nn.Module,
    train_loader,
    g_optimizer,
    d_optimizer,
    latent_dim: int,
    image_dim: int,
    device: torch.device,
    scaler=None,
) -> tuple[float, float, float, float]:
    """Run one training epoch for a conditional GAN.

    Returns avg_d_loss, avg_g_loss, avg_d_real_loss, avg_d_fake_loss.
    G must accept (z, labels) and D must accept (x, labels).
    """
    G.train()
    D.train()

    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = scaler is not None
    d_loss_list, g_loss_list, d_real_loss_list, d_fake_loss_list = [], [], [], []

    for real_images, labels in train_loader:
        real_images = real_images.view(-1, image_dim).to(device, non_blocking=True)
        labels = labels.to(device)
        batch_size = real_images.size(0)

        real_targets = torch.ones(batch_size, 1, device=device)
        fake_targets = torch.zeros(batch_size, 1, device=device)

        # Train Discriminator
        z = torch.randn(batch_size, latent_dim, device=device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            d_real = D(real_images, labels)
            d_fake = D(G(z, labels).detach(), labels)
            d_real_loss = criterion(d_real, real_targets)
            d_fake_loss = criterion(d_fake, fake_targets)
            d_loss = d_real_loss + d_fake_loss
        d_optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(d_loss).backward()
            scaler.step(d_optimizer)
            scaler.update()
        else:
            d_loss.backward()
            d_optimizer.step()

        # Train Generator
        z = torch.randn(batch_size, latent_dim, device=device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            g_loss = criterion(D(G(z, labels), labels), real_targets)
        g_optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(g_loss).backward()
            scaler.step(g_optimizer)
            scaler.update()
        else:
            g_loss.backward()
            g_optimizer.step()

        d_loss_list.append(d_loss.item())
        g_loss_list.append(g_loss.item())
        d_real_loss_list.append(d_real_loss.item())
        d_fake_loss_list.append(d_fake_loss.item())

    return (
        float(np.mean(d_loss_list)),
        float(np.mean(g_loss_list)),
        float(np.mean(d_real_loss_list)),
        float(np.mean(d_fake_loss_list)),
    )


def train_GAN(
    G: nn.Module,
    D: nn.Module,
    train_loader,
    g_optimizer,
    d_optimizer,
    config: dict,
    device: torch.device,
    wandb_kwargs: dict,
    loss_type: str = "bce",
    conditional: bool = False,
) -> None:
    """Full GAN training loop with wandb logging.

    Reads from config: epochs, latent_dim, image_dim.
    loss_type: "bce" or "logistic" — only used when conditional=False.
    conditional=True: calls train_cGAN_epoch and logs one sample per digit class to wandb.
    """
    use_cuda = device.type == "cuda"
    if use_cuda:
        G = torch.compile(G)  # type: ignore[assignment]
        D = torch.compile(D)  # type: ignore[assignment]
    scaler = torch.amp.GradScaler("cuda") if use_cuda else None  # type: ignore[attr-defined]

    epochs = config["epochs"]
    log_img_every = 20

    with wandb.init(**wandb_kwargs) as run:  # type: ignore[arg-type]
        with tqdm(range(epochs), desc="Training", unit="ep") as pbar:
            for epoch in pbar:
                if conditional:
                    avg_d_loss, avg_g_loss, avg_d_real_loss, avg_d_fake_loss = train_cGAN_epoch(
                        G, D, train_loader,
                        g_optimizer, d_optimizer,
                        config["latent_dim"], config["image_dim"], device,
                        scaler=scaler,
                    )
                else:
                    avg_d_loss, avg_g_loss, avg_d_real_loss, avg_d_fake_loss = train_GAN_epoch(
                        G, D, train_loader,
                        g_optimizer, d_optimizer,
                        config["latent_dim"], config["image_dim"], device,
                        loss_type=loss_type, scaler=scaler,
                    )

                pbar.set_postfix(D=f"{avg_d_loss:.4f}", G=f"{avg_g_loss:.4f}")

                run.log({
                    "epoch": epoch + 1,
                    "D_loss": avg_d_loss,
                    "G_loss": avg_g_loss,
                    "D_real_loss": avg_d_real_loss,
                    "D_fake_loss": avg_d_fake_loss,
                }, step=epoch + 1)

                if (epoch + 1) % log_img_every == 0:
                    if conditional:
                        fig = make_cgan_all_digits_figure(G, config["latent_dim"], device, config["num_classes"])
                    else:
                        fig = make_generated_figure(G, config["latent_dim"], device)
                    run.log({"generated_samples": wandb.Image(fig, caption=f"Epoch {epoch + 1}")}, step=epoch + 1)
                    plt.close(fig)


def save_sample(G, epoch, mb_size, z_dim, device):
    out_dir = "out_vanila_GAN2"
    G.eval()
    with torch.no_grad():
        z = torch.randn(mb_size, z_dim).to(device)
        samples = G(z).detach().cpu().numpy()[:16]

    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(f'{out_dir}/{str(epoch).zfill(3)}.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    # Hyperparameters
    X_dim = 784  # 28 x 28
    mb_size = 64
    Z_dim = 1000
    h_dim = 128
    lr = 1e-3

    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    train_dataset = datasets.MNIST(root='../MNIST', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=mb_size, shuffle=True)

    wandb_log = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    G = Generator(Z_dim, h_dim, X_dim).to(device)
    D = Discriminator(X_dim, h_dim).to(device)

    G_solver = optim.Adam(G.parameters(), lr=lr)
    D_solver = optim.Adam(D.parameters(), lr=lr)

    loss_fn = lambda preds, targets: F.binary_cross_entropy(preds, targets)

    if wandb_log:
        wandb.init(project="conditional-gan-mnist")
        wandb.config.update({
            "batch_size": mb_size,
            "Z_dim": Z_dim,
            "X_dim": X_dim,
            "h_dim": h_dim,
            "lr": lr,
        })

    best_g_loss = float('inf')
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    epochs = 100
    for epoch in range(epochs):
        avg_d_loss, avg_g_loss, _, _ = train_GAN_epoch(
            G, D, train_loader, G_solver, D_solver, Z_dim, X_dim, device
        )
        print(f'epoch{epoch}; D_loss: {avg_d_loss:.4f}; G_loss: {avg_g_loss:.4f}')

        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            torch.save(G.state_dict(), os.path.join(save_dir, 'G_best.pth'))
            torch.save(D.state_dict(), os.path.join(save_dir, 'D_best.pth'))
            print(f"Saved Best Models at epoch {epoch} | G_loss: {best_g_loss:.4f}")

        save_sample(G, epoch, mb_size, Z_dim, device)
