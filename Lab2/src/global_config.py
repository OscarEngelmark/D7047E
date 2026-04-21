from pathlib import Path

# Resolved relative to this file, so imports work from any working directory
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR     = PROJECT_DIR / "out"
DATA_DIR    = PROJECT_DIR / "data"
MODELS_DIR  = PROJECT_DIR / "models"
WANDB_DIR   = PROJECT_DIR

for p in [OUT_DIR, MODELS_DIR, WANDB_DIR]:
    p.mkdir(parents=True, exist_ok=True)

WANDB_ENTITY  = "d7047e-group12"
WANDB_PROJECT = "Lab2"
