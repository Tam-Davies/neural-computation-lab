
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.captioning.blip_model import BLIPCaptioner


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/generate_caption.py "
            "path/to/image.jpg"
        )
        return

    image_path = Path(sys.argv[1])

    captioner = BLIPCaptioner()

    result = captioner.generate_caption(image_path)

    output_dir = (
        PROJECT_ROOT / "outputs" / "predictions"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / "image_caption.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nCaption saved to: {output_path}")


if __name__ == "__main__":
    main()
