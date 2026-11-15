
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.config import PROJECT_ROOT, CLASS_NAMES


class ClassifierEvaluator:
    """Evaluate and visualise the aircraft damage classifier."""

    def __init__(self, model=None):
        self.model = model

        self.output_dir = (
            PROJECT_ROOT / "outputs" / "plots"
        )

    def evaluate(self, test_ds):
        """Evaluate model performance on the test dataset."""

        if self.model is None:
            raise ValueError("A trained model is required for evaluation.")

        loss, accuracy = self.model.evaluate(test_ds)

        print(f"\nTest Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        y_true = []
        y_pred = []

        for images, labels in test_ds:
            predictions = self.model.predict(
                images,
                verbose=0
            )

            y_true.extend(labels.numpy().flatten())

            y_pred.extend(
                (predictions.flatten() >= 0.5).astype(int)
            )

        print("\nClassification Report:")

        print(
            classification_report(
                y_true,
                y_pred,
                target_names=CLASS_NAMES,
                zero_division=0,
            )
        )

        self.plot_confusion_matrix(y_true, y_pred)

        return y_true, y_pred

    def plot_confusion_matrix(self, y_true, y_pred):
        """Generate and save the confusion matrix."""

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        matrix = confusion_matrix(
            y_true,
            y_pred
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=CLASS_NAMES,
        )

        display.plot(
            cmap="Blues",
            values_format="d"
        )

        plt.title("Aircraft Damage Classification")
        plt.tight_layout()

        output_path = (
            self.output_dir / "confusion_matrix.png"
        )

        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Confusion matrix saved: {output_path}")

    def plot_training_history(self, history):
        """Plot training and validation accuracy and loss."""

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Accuracy curve
        plt.figure(figsize=(8, 5))

        plt.plot(
            history["accuracy"],
            label="Training"
        )

        plt.plot(
            history["val_accuracy"],
            label="Validation"
        )

        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            self.output_dir / "accuracy_curve.png",
            dpi=300
        )

        plt.close()

        # Loss curve
        plt.figure(figsize=(8, 5))

        plt.plot(
            history["loss"],
            label="Training"
        )

        plt.plot(
            history["val_loss"],
            label="Validation"
        )

        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            self.output_dir / "loss_curve.png",
            dpi=300
        )

        plt.close()

        print("Training history plots saved successfully.")
