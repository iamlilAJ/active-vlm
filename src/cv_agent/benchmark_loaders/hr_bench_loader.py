import base64
import io
from typing import Any

import structlog
from datasets import load_dataset
from PIL import Image

from .base import BaseDatasetLoader

logger = structlog.get_logger(__name__)


class HRBenchLoader(BaseDatasetLoader):
    """Loads the HR-Bench dataset from the Hugging Face hub."""

    def __init__(self, split_name: str = "hrbench_4k"):
        """
        Initializes the loader.
        Args:
            split_name: The name of the split to load (e.g., "hrbench_4k", "hrbench_8k").
        """
        self.split_name = split_name
        try:
            # Load the specific configuration and split
            self.dataset = load_dataset(
                "DreamMr/HR-Bench", name="hrbench_version_split", split=self.split_name
            )
        except Exception as e:
            logger.error("Failed to load 'DreamMr/HR-Bench' from Hugging Face: %s", e)
            raise

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Fetches and maps a sample from the HR-Bench dataset."""
        example = self.dataset[idx]

        image_b64_string = example["image"]
        image_bytes = base64.b64decode(image_b64_string)
        image: Image.Image = Image.open(io.BytesIO(image_bytes))
        question_text: str = example["question"]
        correct_answer: str = example["answer"]

        # 1. Use the correct 'category' key for the task_name
        task_name: str = example["category"]

        # 2. Build the options list from the 'A', 'B', 'C', 'D' columns
        options_list = [
            f"(A) {example['A']}",
            f"(B) {example['B']}",
            f"(C) {example['C']}",
            f"(D) {example['D']}",
        ]

        sample_id = f"{self.split_name}_{idx}"  # Create a unique ID

        # Combine the question and options
        full_question = question_text + "\n" + "\n".join(options_list)

        return {
            "image": image,
            "question": full_question,
            "correct_answer": correct_answer,
            "task_name": task_name,
            "sample_id": sample_id,
        }
