
import tarfile
import urllib.request
from pathlib import Path

from src.config import (
    ARCHIVE_PATH,
    DATASET_DIR,
    DATASET_URL,
    RAW_DATA_DIR,
    CLASS_NAMES,
)


class AircraftDataset:
    """Download, extract, and validate the aircraft damage dataset."""

    def __init__(self):
        self.archive_path = ARCHIVE_PATH
        self.dataset_dir = DATASET_DIR
        self.raw_dir = RAW_DATA_DIR

        self.splits = ("train", "valid", "test")

    def download(self):
        """Download the dataset archive if it is not already available."""

        self.raw_dir.mkdir(parents=True, exist_ok=True)

        if self.archive_path.exists():
            print("Dataset archive already exists.")
            return

        print("Downloading aircraft damage dataset...")

        urllib.request.urlretrieve(
            DATASET_URL,
            self.archive_path
        )

        print("Download completed.")

    def extract(self):
        """Extract the dataset archive safely."""

        if self.dataset_dir.exists():
            print("Dataset directory already exists.")
            return

        if not self.archive_path.exists():
            raise FileNotFoundError(
                "Dataset archive not found. Run download() first."
            )

        with tarfile.open(self.archive_path, "r:*") as archive:
            destination = self.raw_dir.resolve()

            for member in archive.getmembers():
                target = (self.raw_dir / member.name).resolve()

                if not target.is_relative_to(destination):
                    raise ValueError(
                        f"Unsafe archive path detected: {member.name}"
                    )

            archive.extractall(self.raw_dir)

        print("Dataset extraction completed.")

    def validate(self):
        """Check the expected dataset directory structure."""

        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dataset_dir}"
            )

        for split in self.splits:
            split_dir = self.dataset_dir / split

            if not split_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing dataset split: {split_dir}"
                )

            for class_name in CLASS_NAMES:
                class_dir = split_dir / class_name

                if not class_dir.is_dir():
                    raise FileNotFoundError(
                        f"Missing class directory: {class_dir}"
                    )

        print("Dataset structure validated successfully.")

    def prepare(self):
        """Download, extract, and validate the dataset."""

        self.download()
        self.extract()
        self.validate()
