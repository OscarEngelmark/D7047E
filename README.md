# D7047E – Advanced Deep Learning Labs

## Lab 0: Introduction to Deep Learning with PyTorch

Lab 0 covers foundational deep learning concepts through three experiments on image classification, using CIFAR-10, MNIST, and SVHN datasets.

### Files

#### [Lab0_01.ipynb](Lab0/src/Lab0_01.ipynb) – Optimizer & Activation Function Comparison

Trains a custom `SimpleCNN` on CIFAR-10 (10-class image classification, 32×32 RGB) comparing the effect of optimizer and activation function choices.

| Experiment | Optimizer | Activation | Test Acc |
|------------|-----------|------------|----------|
| A | SGD | LeakyReLU | 63.23% |
| B | Adam | LeakyReLU | 71.14% |
| C | Adam | Tanh | 73.20% |

**Key takeaway:** Adam outperforms SGD, and Tanh slightly outperforms LeakyReLU on this task.

---

#### [Lab0_02.ipynb](Lab0/src/Lab0_02.ipynb) – Transfer Learning from ImageNet (AlexNet)

Uses a pretrained AlexNet (ImageNet) adapted for CIFAR-10 classification. Compares two transfer learning strategies.

| Experiment | Strategy | Test Acc |
|------------|----------|----------|
| A | Full fine-tuning | 90.35% |
| B | Feature extraction (frozen backbone) | 83.06% |

**Key takeaway:** Fine-tuning all layers substantially outperforms using a frozen backbone, since CIFAR-10 differs from ImageNet.

---

#### [Lab0_03.ipynb](Lab0/src/Lab0_03.ipynb) – Cross-Domain Transfer: MNIST → SVHN

Investigates transfer learning across domains using the same `SimpleCNN` architecture. The model is first trained on MNIST (clean grayscale digits), then evaluated on SVHN (real-world street-view digits).

| Experiment | Description | Test Acc |
|------------|-------------|----------|
| A | Train on MNIST | 99.43% |
| B | Zero-shot evaluation on SVHN | 29.92% |
| C | Fine-tune on SVHN | 90.15% |

**Key takeaway:** The large domain gap between synthetic (MNIST) and real-world (SVHN) data leads to poor zero-shot transfer. Fine-tuning recovers strong performance.

---

#### [utils.py](Lab0/src/utils.py) – Shared Training Utilities

Reusable module used across all notebooks. Provides:

- `make_loaders()` – Splits a dataset into train/val/test `DataLoader`s with a configurable split ratio.
- `train()` – Runs one training epoch; returns loss and accuracy.
- `validate()` – Runs one evaluation epoch (no gradients); returns loss and accuracy.
- `fit()` – Full training loop with best-checkpoint restoration and optional [Weights & Biases](https://wandb.ai) logging.
- `evaluate()` – Evaluates a trained model on the test set and prints results.
