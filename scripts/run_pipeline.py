
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import AircraftDamagePipeline


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/run_pipeline.py "
            "path/to/aircraft_image.jpg"
        )
        return

    image_path = Path(sys.argv[1])

    pipeline = AircraftDamagePipeline()

    result = pipeline.analyze(image_path)

    print("\n" + "=" * 50)
    print("FINAL AIRCRAFT DAMAGE ANALYSIS")
    print("=" * 50)

    print(
        "Predicted damage:",
        result["damage_classification"]["predicted_class"]
    )

    print(
        "Classification confidence:",
        f'{result["damage_classification"]["confidence"]}%'
    )

    print(
        "Image caption:",
        result["image_caption"]
    )

    # Save combined result
    output_dir = (
        PROJECT_ROOT / "outputs" / "predictions"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / "combined_analysis.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nCombined report saved to: {output_path}")


if __name__ == "__main__":
    main()
