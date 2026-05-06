from pathlib import Path

# All paths resolve relative to this file so imports work from any directory.
PROJECT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_DIR / "data" / "flickr8k"
IMAGE_DIR     = DATA_DIR / "Images"
CAPTIONS_FILE = DATA_DIR / "captions.txt"
RUNS_DIR      = PROJECT_DIR / "runs"
MODELS_DIR    = PROJECT_DIR / "models"

WANDB_ENTITY  = "d7047e-group12"
WANDB_PROJECT = "Lab3"
