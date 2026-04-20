from pathlib import Path

# Resolved relative to this file, so imports work from any working directory
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR     = PROJECT_DIR / "out"
MODELS_DIR  = PROJECT_DIR / "Models"
WANDB_DIR   = PROJECT_DIR

WANDB_ENTITY  = "d7047e-group12"
WANDB_PROJECT = "Lab2"
