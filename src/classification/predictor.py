
from pathlib import Path

import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

from src.config import (
    PROJECT_ROOT,
    IMAGE_SIZE,
    CLASS_NAMES,
)


class AircraftPredictor:
    """Predict aircraft damage from a single image."""

    def __init__(self, model_path=None):
        self.model_path = model_path or (
            PROJECT_ROOT
            / "models"
            / "classification"
            / "vgg16_aircraft_damage.keras"
        )

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Trained model not found: {self.model_path}"
            )

        self.model = tf.keras.models.load_model(
            self.model_path
        )

    def preprocess_image(self, image_path):
        """Load and preprocess an image for VGG16."""

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        image = image.resize(IMAGE_SIZE)

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        image_array = preprocess_input(image_array)

        return image_array

    def predict(self, image_path):
        """Predict whether an aircraft image contains a crack or dent."""

        image_array = self.preprocess_image(image_path)

        probability = float(
            self.model.predict(
                image_array,
                verbose=0
            )[0][0]
        )

        predicted_index = int(probability >= 0.5)

        predicted_class = CLASS_NAMES[predicted_index]

        confidence = (
            probability
            if predicted_index == 1
            else 1 - probability
        )

        result = {
            "image": str(image_path),
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "probability": round(probability, 4),
        }

        print("\nAircraft Damage Prediction")
        print("--------------------------")
        print(f"Image: {result['image']}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {result['confidence']}%")

        return result
