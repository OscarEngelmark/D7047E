# D7047E – Advanced Deep Learning Labs

## Lab 0: Introduction to Deep Learning with PyTorch

Lab 0 covers foundational deep learning concepts through three experiments on image classification, using CIFAR-10, MNIST, and SVHN datasets.

### Files

#### [Lab0_01.ipynb](Lab0/src/Lab0_01.ipynb) – Optimizer & Activation Function Comparison

Trains a custom `SimpleCNN` on CIFAR-10 (10-class image classification, 32×32 RGB) comparing the effect of optimizer and activation function choices.

---

#### [Lab0_02.ipynb](Lab0/src/Lab0_02.ipynb) – Transfer Learning from ImageNet (AlexNet)

Uses a pretrained AlexNet (ImageNet) adapted for CIFAR-10 classification. Compares two transfer learning strategies.

---

#### [Lab0_03.ipynb](Lab0/src/Lab0_03.ipynb) – Cross-Domain Transfer: MNIST → SVHN

Investigates transfer learning across domains using the same `SimpleCNN` architecture. The model is first trained on MNIST (clean grayscale digits), then evaluated on SVHN (real-world street-view digits).

---

#### [utils.py](Lab0/src/utils.py) – Shared Training Utilities

Reusable module used across all notebooks. Provides:

- `make_loaders()` – Splits a dataset into train/val/test `DataLoader`s with a configurable split ratio.
- `train()` – Runs one training epoch; returns loss and accuracy.
- `validate()` – Runs one evaluation epoch (no gradients); returns loss and accuracy.
- `fit()` – Full training loop with best-checkpoint restoration and optional [Weights & Biases](https://wandb.ai) logging.
- `evaluate()` – Evaluates a trained model on the test set and prints results.

---

## Lab 1: Sentiment Analysis with Classical and Transformer Models

Lab 1 benchmarks four sentiment classifiers — a feedforward ANN, a BiLSTM, DistilBERT, and RoBERTa — across three Amazon review datasets of increasing scale and task complexity.

### Datasets

| File | Reviews | Task | Classes | Source |
|------|---------|------|---------|--------|
| `amazon_cells_labelled.txt` | 1,000 | Binary sentiment | Negative / Positive | Course material |
| `amazon_cells_labelled_LARGE_25K.txt` | 25,000 | Binary sentiment | Negative / Positive | Course material |
| `Video_Games.json` | ~2.56 M | Star-rating prediction | 1 – 5 stars | Ni et al. (2019) |

The Video Games dataset is sourced from the [Amazon Review Dataset](https://nijianmo.github.io/amazon/index.html) (Ni et al., 2019). The smaller cell-phone review files were provided as course material.

Each dataset is split into train / validation / test sets (81% / 9% / 10%) using a stratified split to preserve class balance. Splits are written to `Lab1/data/splits/` by [Lab1_Splitting.ipynb](Lab1/src/Lab1_Splitting.ipynb) and loaded by all downstream notebooks.

> Ni, J., Li, J., & McAuley, J. (2019). Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)* (pp. 188–197). https://doi.org/10.18653/v1/D19-1018

### Files

#### [Lab1_Splitting.ipynb](Lab1/src/Lab1_Splitting.ipynb) – Data Preparation

Single source of truth for data loading and splitting. Applies a reproducible stratified split (seed = 1) to all three datasets and saves the nine resulting CSVs to `Lab1/data/splits/`. Must be run once before any experiment notebook.

---

#### [Lab1_ANN.ipynb](Lab1/src/Lab1_ANN.ipynb) – Feedforward ANN (TF-IDF + SVD)

Classical NLP pipeline: text is lowercased and cleaned, vectorized with TF-IDF, compressed with Truncated SVD (LSA, 300 components), and fed into a three-layer feedforward network with BatchNorm and Dropout. Trained with Adam and cross-entropy loss.

---

#### [Lab1_BiLSTM.ipynb](Lab1/src/Lab1_BiLSTM.ipynb) – Bidirectional LSTM

Sequence model that learns token embeddings from scratch. Text is lightly cleaned (stopwords kept, as they provide positional context), tokenized via a project vocabulary, and passed through a BiLSTM (64 → 128 hidden units). The concatenated forward/backward final states feed a linear classifier.

---

#### [Lab1_DistilBERT.ipynb](Lab1/src/Lab1_DistilBERT.ipynb) – DistilBERT Fine-Tuning

Fine-tunes `distilbert-base-uncased` for sequence classification using the HuggingFace `Trainer`. Uses AdamW with a linear warmup schedule and mixed-precision training. For the imbalanced Video Games split (~58% five-star), inverse-frequency class weights and label smoothing (0.1) are applied via a `WeightedTrainer`.

---

#### [Lab1_RoBERTa.ipynb](Lab1/src/Lab1_RoBERTa.ipynb) – RoBERTa Fine-Tuning

Same training setup as the DistilBERT notebook but uses `roberta-base`, a larger and more robustly pre-trained model.

---

#### [Lab1_Comparison.ipynb](Lab1/src/Lab1_Comparison.ipynb) – Cross-Model Comparison

Side-by-side evaluation of all four models across all three datasets. Loads saved checkpoints, runs inference, and reports per-model classification reports, accuracy / macro F1 / weighted F1 comparison tables, and a 4-panel confusion-matrix grid for each dataset.

---

#### [data_loading_code.py](Lab1/src/data_loading_code.py) – Text Preprocessing

Provides `preprocess_pandas()`, which applies TF-IDF-oriented cleaning to a DataFrame: lowercasing, removal of emails/IPs, punctuation stripping, digit removal, and stopword filtering.

---

#### [transformer_utils.py](Lab1/src/transformer_utils.py) – Transformer Training Utilities

Shared helpers for the DistilBERT and RoBERTa notebooks:

- `build_tf_datasets()` – Tokenizes train/val/test DataFrames into HuggingFace `Dataset` objects.
- `compute_metrics_tf()` – Returns accuracy, macro F1, and weighted F1 for `Trainer` eval callbacks.
- `WeightedTrainer` – `Trainer` subclass that applies inverse-frequency class weights to the loss, used for the imbalanced Video Games split.
- `evaluate_tf()` – Runs inference with a `Trainer`, prints a classification report, and returns predictions.
- `plot_confusion_matrix_tf()` – Plots a labeled confusion matrix (optionally normalized).

---

#### [utils.py](Lab1/src/utils.py) – Shared Training Utilities

Extends the Lab 0 utility set with Lab 1-specific helpers:

- `stratified_split()` – Stratified train/val/test split preserving class balance.
- `train()` / `validate()` – Epoch-level training and evaluation loops; `validate` additionally returns raw predictions for confusion matrix plotting.
- `fit()` – Full training loop with best-checkpoint saving and optional W&B logging.
- `evaluate()` / `plot_confusion_matrix()` – Test-set evaluation and visualization.
- `load_ann_run()` / `load_bilstm_run()` – Reload a saved ANN or BiLSTM checkpoint together with its fitted vectorizer/vocabulary for inference in the comparison notebook.

---

## Lab 2: Generative Models – GANs, Adversarial Attacks & Diffusion

Lab 2 explores generative image modeling on MNIST across five tasks: vanilla and logistic-loss GANs, a conditional GAN for class-conditioned generation, adversarial attacks on a CNN classifier, and a simple diffusion model.

### Files

#### [Lab2_Task1.ipynb](Lab2/src/Lab2_Task1.ipynb) – Vanilla GAN

Trains a two-layer feedforward GAN (Generator + Discriminator) on MNIST. Uses `BCEWithLogitsLoss` on raw logits for numerical stability and AMP compatibility. Default hyperparameters: `latent_dim=100`, `h_dim=128`, `lr=1e-3`, `batch_size=256`. Training is logged to Weights & Biases.

---

#### [Lab2_Task2.ipynb](Lab2/src/Lab2_Task2.ipynb) – Logistic Loss GAN

Same architecture as Task 1 but with softplus-based logistic loss: D loss = `softplus(-f_real) + softplus(f_fake)`, G loss = `softplus(-f_fake)`. Trains for 5, 10, and 50 epochs and compares outputs to the BCE variant. Mathematically equivalent to `BCEWithLogitsLoss`; the task motivates understanding the loss formulation.

---

#### [Lab2_Task3.ipynb](Lab2/src/Lab2_Task3.ipynb) – Conditional GAN (CGAN)

Extends the vanilla GAN with class conditioning. `ConditionalGenerator` and `ConditionalDiscriminator` each accept raw integer labels, which are one-hot encoded internally via `F.one_hot`. W&B image logging uses a 2×5 grid with one generated sample per digit class (0–9). Trained with `BCEWithLogitsLoss` and AMP.

---

#### [Lab2_Task4.ipynb](Lab2/src/Lab2_Task4.ipynb) – Adversarial Attacks

Trains a CNN classifier on MNIST, then crafts adversarial examples via gradient-based perturbation to misclassify 4s as 9s. Also feeds pure random noise through the classifier and compares its confidence to that of the adversarial examples.

---

#### [Lab2_Task5.ipynb](Lab2/src/Lab2_Task5.ipynb) – Simple Diffusion Model

Notebook by [Fareed Khan](https://github.com/FareedKhan-dev). Builds a score-based diffusion model for MNIST from scratch, progressing through three U-Net variants: skip connections via concatenation, skip connections via addition, and a transformer-augmented U-Net with cross-attention for class conditioning. Uses an Euler–Maruyama sampler and exponential noise schedule (σ=25). Once trained, the model generates samples for a specified digit class (0–9).

---

#### [GAN.py](Lab2/src/GAN.py) – GAN Models & Training Logic

Central module for all GAN tasks. Provides:

- `Generator` / `Discriminator` – Two-layer feedforward networks with Xavier initialization and raw logit output.
- `ConditionalGenerator` / `ConditionalDiscriminator` – Label-conditioned variants; one-hot encoding applied internally.
- `train_GAN_epoch()` – One unconditional training epoch; supports `loss_type="bce"` or `"logistic"`.
- `train_cGAN_epoch()` – One conditional training epoch with label-aware model calls.
- `train_GAN()` – Full training loop: applies `torch.compile` and AMP on CUDA, logs losses and generated images to W&B every `epochs // 20` steps.

---

#### [utils.py](Lab2/src/utils.py) – Visualization & Checkpoint Helpers

- `device_check()` – Detects the best available device (CUDA → MPS → CPU) and prints environment info.
- `make_generated_figure()` – Samples 16 images from a generator and returns a 4×4 matplotlib figure.
- `make_cgan_all_digits_figure()` – Generates one sample per digit class (0–9) and returns a 2×5 labeled figure; used for W&B logging during CGAN training.
- `make_cgan_figure()` – Generates a 4×4 grid for a single specified digit class.
- `show_generated_images()` / `show_cgan_images()` – Notebook display wrappers.
- `save_generated_grid()` – Saves a generated image grid to disk.
- `build_model_name()` – Constructs a readable checkpoint filename from a training config dict.

---

#### [global_config.py](Lab2/src/global_config.py) – Shared Constants

Defines shared paths (`DATA_DIR`, `MODELS_DIR`, `WANDB_DIR`) and W&B entity/project used across all Lab 2 notebooks.
