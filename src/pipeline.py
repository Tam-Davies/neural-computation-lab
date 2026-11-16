
from pathlib import Path

from src.config import PROJECT_ROOT
from src.classification.predictor import AircraftPredictor
from src.captioning.blip_model import BLIPCaptioner


class AircraftDamagePipeline:
    """Combine aircraft damage classification and image captioning."""

    def __init__(self):
        print("Initializing aircraft damage pipeline...")

        self.classifier = AircraftPredictor()
        self.captioner = BLIPCaptioner()

        print("Pipeline initialized successfully.")

    def analyze(self, image_path):
        """Classify aircraft damage and generate an image caption."""

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        print("\nRunning aircraft image analysis...")

        # Damage classification
        classification = self.classifier.predict(image_path)

        # Image captioning
        caption = self.captioner.generate_caption(image_path)

        result = {
            "image": str(image_path),
            "damage_classification": {
                "predicted_class": classification["predicted_class"],
                "confidence": classification["confidence"],
                "probability": classification["probability"],
            },
            "image_caption": caption["caption"],
        }

        return result
