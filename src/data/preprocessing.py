
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import preprocess_input

from src.config import (
    DATASET_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
    CLASS_NAMES,
)


class ImagePreprocessor:
    """Load and preprocess aircraft damage images for VGG16."""

    def __init__(self):
        self.dataset_dir = DATASET_DIR
        self.image_size = IMAGE_SIZE
        self.batch_size = BATCH_SIZE
        self.seed = RANDOM_SEED

    def load_split(self, split, shuffle=False):
        """Load one dataset split from its directory."""

        split_dir = self.dataset_dir / split

        dataset = tf.keras.utils.image_dataset_from_directory(
            split_dir,
            class_names=CLASS_NAMES,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode="int",
            shuffle=shuffle,
            seed=self.seed if shuffle else None,
        )

        return dataset

    @staticmethod
    def preprocess_batch(images, labels):
        """Apply VGG16 preprocessing to an image batch."""

        images = preprocess_input(images)
        return images, labels

    def prepare(self):
        """Load and preprocess all three dataset splits."""

        train_ds = self.load_split("train", shuffle=True)
        valid_ds = self.load_split("valid")
        test_ds = self.load_split("test")

        autotune = tf.data.AUTOTUNE

        train_ds = (
            train_ds
            .map(self.preprocess_batch, num_parallel_calls=autotune)
            .prefetch(autotune)
        )

        valid_ds = (
            valid_ds
            .map(self.preprocess_batch, num_parallel_calls=autotune)
            .prefetch(autotune)
        )

        test_ds = (
            test_ds
            .map(self.preprocess_batch, num_parallel_calls=autotune)
            .prefetch(autotune)
        )

        return train_ds, valid_ds, test_ds
