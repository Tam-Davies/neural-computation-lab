
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf

from src.data.preprocessing import ImagePreprocessor
from src.classification.evaluator import ClassifierEvaluator


def main():
    model_path = (
        PROJECT_ROOT
        / "models"
        / "classification"
        / "vgg16_aircraft_damage.keras"
    )

    model = tf.keras.models.load_model(model_path)

    _, _, test_ds = ImagePreprocessor().prepare()

    evaluator = ClassifierEvaluator(model)
    evaluator.evaluate(test_ds)



if __name__ == "__main__":
    main()
