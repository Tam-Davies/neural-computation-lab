
import sys
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.model import VGG16Classifier
from src.classification.trainer import ClassifierTrainer
from src.classification.evaluator import ClassifierEvaluator
from src.data.preprocessing import ImagePreprocessor


def main():
    # Load datasets
    train_ds, valid_ds, test_ds = ImagePreprocessor().prepare()

    # Build and compile model
    classifier = VGG16Classifier()

    model = classifier.build()
    classifier.compile()

    # Train model
    trainer = ClassifierTrainer(model)

    history = trainer.train(
        train_ds,
        valid_ds
    )

    # Save trained model
    trainer.save()

    # Save training history
    history_dir = (
        PROJECT_ROOT / "outputs" / "predictions"
    )

    history_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    history_path = history_dir / "training_history.pkl"

    with open(history_path, "wb") as file:
        pickle.dump(history.history, file)

    print(f"Training history saved: {history_path}")

    # Generate training history plots
    evaluator = ClassifierEvaluator(model)

    evaluator.plot_training_history(
        history.history
    )

    # Final training results
    print("\nTraining completed.")

    print(
        "Final training accuracy:",
        history.history["accuracy"][-1]
    )

    print(
        "Final validation accuracy:",
        history.history["val_accuracy"][-1]
    )


if __name__ == "__main__":
    main()
