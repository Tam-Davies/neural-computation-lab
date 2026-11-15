
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from src.config import IMAGE_SIZE, LEARNING_RATE, DROPOUT_RATE


class VGG16Classifier:
    """VGG16 feature extractor with a custom binary classifier."""

    def __init__(self):
        self.model = None
        self.base_model = None

    def build(self):
        """Build the VGG16 transfer-learning architecture."""

        self.base_model = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(*IMAGE_SIZE, 3),
        )

        output = self.base_model.layers[-1].output
        output = Flatten()(output)

        self.base_model = Model(
            inputs=self.base_model.input,
            outputs=output,
        )

        # Freeze pretrained VGG16 layers
        for layer in self.base_model.layers:
            layer.trainable = False

        self.model = Sequential([
            self.base_model,
            Dense(512, activation="relu"),
            Dropout(DROPOUT_RATE),
            Dense(512, activation="relu"),
            Dropout(DROPOUT_RATE),
            Dense(1, activation="sigmoid"),
        ])

        return self.model

    def compile(self):
        """Compile the classifier using IBM's specified settings."""

        if self.model is None:
            raise RuntimeError("Build the model before compiling it.")

        self.model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        return self.model
