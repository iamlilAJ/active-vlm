from abc import ABC, abstractmethod
from typing import Any


class BaseDatasetLoader(ABC):
    """
    Abstract base class for a dataset loader.
    Defines the interface for __len__ and __getitem__
    to provide a unified data structure.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Fetches a single sample by its index and returns it in a
        standardized dictionary format.

        Standardized Format:
        {
            "image": PIL.Image.Image,
            "question": str,
            "correct_answer": str,
            "task_name": str,
            "sample_id": str  # Must be unique (e.g., "mme_123")
        }
        """
        pass

    def get_metadata(self, idx: int) -> dict[str, Any]:
        """
        Get cheap metadata without loading images.

        This method allows filtering and preprocessing logic to access
        metadata (task_name, sample_id) without triggering expensive
        image I/O operations.

        Default implementation falls back to __getitem__, which is slow
        but correct. Subclasses should override this method to avoid
        loading images when only metadata is needed.

        Returns:
            dict: Metadata with at minimum {"task_name": str, "sample_id": str}

        Raises:
            KeyError: If the loader's __getitem__ does not return required fields
        """
        sample = self[idx]
        if "task_name" not in sample:
            raise KeyError(
                f"Loader {self.__class__.__name__} did not return 'task_name' for sample {idx}"
            )
        if "sample_id" not in sample:
            raise KeyError(
                f"Loader {self.__class__.__name__} did not return 'sample_id' for sample {idx}"
            )
        return {
            "task_name": sample["task_name"],
            "sample_id": sample["sample_id"],
        }
