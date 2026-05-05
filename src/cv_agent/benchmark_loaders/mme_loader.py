import json
import os
from typing import Any

import structlog
from PIL import Image

from .base import BaseDatasetLoader

logger = structlog.get_logger(__name__)


class MMELoader(BaseDatasetLoader):
    """Loads the MME-RealWorld-Lite dataset from local JSON and image files."""

    def __init__(
        self,
        data_path: str | None = None,
        image_dir: str | None = None,
    ):
        self.data_path = data_path
        self.image_dir = image_dir

        if not data_path or not image_dir:
            raise ValueError("MME loader requires data_path and image_dir arguments.")

        try:
            with open(data_path) as f:
                self.dataset = json.load(f)
        except Exception as e:
            logger.error(
                "dataset_load_failed", dataset="MME", data_path=str(data_path), error=str(e)
            )
            raise

    def __len__(self) -> int:
        return len(self.dataset)

    def _extract_metadata(self, idx: int) -> dict[str, str]:
        """Extract metadata fields from dataset entry."""
        example = self.dataset[idx]
        return {
            "task_name": f"{example['Task']}/{example['Subtask']}",
            "sample_id": f"mme_{idx}",
        }

    def get_metadata(self, idx: int) -> dict[str, Any]:
        """Get metadata without loading images from disk."""
        return self._extract_metadata(idx)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Fetches and maps a sample from the MME dataset."""
        metadata = self._extract_metadata(idx)
        example = self.dataset[idx]

        # Load image from file path
        image_file = example["Image"]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path)
        image.load()  # Force load into memory, releases file descriptor

        # Build the full prompt/question
        question = example["Text"]
        answer_choices = example["Answer choices"]
        prompt = question + "\n" + "\n".join(answer_choices)

        correct_answer = example["Ground truth"]

        return {
            "image": image,
            "question": prompt,
            "correct_answer": correct_answer,
            **metadata,  # Single source of truth
        }
