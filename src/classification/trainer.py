from pathlib import Path

from src.config import EPOCHS, PROJECT_ROOT


class ClassifierTrainer:
    """Train and save the VGG16 aircraft damage classifier."""

    def __init__(self, model):
        self.model = model
        self.history = None

        self.model_dir = (
            PROJECT_ROOT / "models" / "classification"
        )

    def train(self, train_ds, valid_ds):
        """Train the model using training and validation datasets."""

        self.history = self.model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCHS,
        )

        return self.history

    def save(self, filename="vgg16_aircraft_damage.keras"):
        """Save the trained model."""

        self.model_dir.mkdir(parents=True, exist_ok=True)

        model_path = self.model_dir / filename
        self.model.save(model_path)

        print(f"Model saved to: {model_path}")

        return model_path