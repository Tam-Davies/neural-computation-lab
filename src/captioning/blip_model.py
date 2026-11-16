
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
)


class BLIPCaptioner:
    """Generate image captions using pretrained BLIP."""

    MODEL_NAME = "Salesforce/blip-image-captioning-base"

    def __init__(self):
        self.device = torch.device("cpu")

        print(f"Loading BLIP model: {self.MODEL_NAME}")

        self.processor = BlipProcessor.from_pretrained(
            self.MODEL_NAME
        )

        self.model = BlipForConditionalGeneration.from_pretrained(
            self.MODEL_NAME
        )

        self.model.to(self.device)
        self.model.eval()

        print("BLIP model loaded successfully.")

    def load_image(self, image_path):
        """Load an image and convert it to RGB."""

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        return Image.open(image_path).convert("RGB")

    def generate_caption(self, image_path, max_new_tokens=50):
        """Generate a natural-language caption for an image."""

        image = self.load_image(image_path)

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=3,
            )

        caption = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=True
        )

        result = {
            "image": str(image_path),
            "caption": caption.strip(),
        }

        print("\nBLIP Image Caption")
        print("------------------")
        print(f"Image: {result['image']}")
        print(f"Caption: {result['caption']}")

        return result
