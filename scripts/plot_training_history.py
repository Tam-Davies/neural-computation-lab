
import sys
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.evaluator import ClassifierEvaluator


def main():
    history_path = (
        PROJECT_ROOT
        / "outputs"
        / "predictions"
        / "training_history.pkl"
    )

    if not history_path.exists():
        raise FileNotFoundError(
            f"Training history not found: {history_path}"
        )

    with open(history_path, "rb") as file:
        history = pickle.load(file)

    evaluator = ClassifierEvaluator()
    evaluator.plot_training_history(history)


if __name__ == "__main__":
    main()
