from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset
DATASET_NAME = "aircraft_damage_dataset_v1"
DATASET_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar"
)

DATASET_DIR = RAW_DATA_DIR / DATASET_NAME
ARCHIVE_PATH = RAW_DATA_DIR / f"{DATASET_NAME}.tar"

# Classification configuration
CLASS_NAMES = ["crack", "dent"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
RANDOM_SEED = 42

# Model configuration
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.3