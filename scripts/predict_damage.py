import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.predictor import AircraftPredictor


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/predict_damage.py "
            "path/to/aircraft_image.jpg"
        )
        return

    image_path = Path(sys.argv[1])

    predictor = AircraftPredictor()

    predictor.predict(image_path)


if __name__ == "__main__":
    main()
