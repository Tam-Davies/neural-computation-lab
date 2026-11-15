import sys
from pathlib import Path

# Add repository root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.model import VGG16Classifier
from src.classification.trainer import ClassifierTrainer
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
    history = trainer.train(train_ds, valid_ds)

    # Save model
    trainer.save()

    print("\nTraining completed.")
    print("Final training accuracy:", history.history["accuracy"][-1])
    print("Final validation accuracy:", history.history["val_accuracy"][-1])


if __name__ == "__main__":
    main()