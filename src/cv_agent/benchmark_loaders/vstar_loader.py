import json
import os
from typing import Any

import structlog
from PIL import Image

from .base import BaseDatasetLoader

logger = structlog.get_logger(__name__)


class VStarLoader(BaseDatasetLoader):
    """
    Loads the V*STAR benchmark dataset from local .jsonl and image files.
    """

    def __init__(
        self,
        data_path: str | None = None,
        image_dir: str | None = None,
    ):
        self.data_path = data_path
        self.image_dir = image_dir

        if not self.data_path or not self.image_dir:
            raise ValueError("VStarLoader requires both --data_path and --image_dir.")

        try:
            # --- Load from local JSONL file ---
            logger.info(
                "dataset_loading", dataset="V*STAR", data_path=str(self.data_path), source="local"
            )
            self.dataset = []
            with open(self.data_path) as f:
                for line in f:
                    self.dataset.append(json.loads(line))

        except Exception as e:
            logger.error("dataset_load_failed", dataset="V*STAR", error=str(e))
            raise

    def __len__(self) -> int:
        return len(self.dataset)

    def _extract_metadata(self, idx: int) -> dict[str, str]:
        """Extract metadata fields from dataset entry."""
        example = self.dataset[idx]
        return {
            "task_name": example["category"],
            "sample_id": f"vstar_{example['question_id']}",
        }

    def get_metadata(self, idx: int) -> dict[str, Any]:
        """Get metadata without loading images from disk."""
        return self._extract_metadata(idx)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Fetches and maps a sample from the V*STAR dataset."""
        metadata = self._extract_metadata(idx)
        example = self.dataset[idx]

        # --- Handle local image loading ---
        image_data = example["image"]

        # image_data is a string path like "direct_attributes/sa_4690.jpg"
        image_path = os.path.join(self.image_dir, image_data)
        image = Image.open(image_path)
        image.load()  # Force load into memory, releases file descriptor

        prompt = example["text"]
        correct_answer = example["label"]  # e.g., "A"

        return {
            "image": image,
            "question": prompt,
            "correct_answer": correct_answer,
            **metadata,  # Single source of truth
        }
