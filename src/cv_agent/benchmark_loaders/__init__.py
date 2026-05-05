import structlog

from .base import BaseDatasetLoader
from .cv_bench_loader import CVBenchLoader
from .hr_bench_loader import HRBenchLoader
from .mme_loader import MMELoader
from .vstar_loader import VStarLoader

logger = structlog.get_logger(__name__)


def get_dataset_loader(name: str, **kwargs) -> BaseDatasetLoader:
    """
    Factory function to get the specified dataset loader.

    Args:
        name: Dataset name
        **kwargs: Additional arguments to pass to the loader constructor.
                 For example: data_path, image_dir for MMELoader and VStarLoader
    """
    if name == "mme":
        loader = MMELoader(**kwargs)
    elif name == "cvbench":
        loader = CVBenchLoader(**kwargs)
    elif name == "vstar":
        loader = VStarLoader(**kwargs)
    elif name == "hrbench-4k":
        loader = HRBenchLoader(split_name="hrbench_4k", **kwargs)
    elif name == "hrbench-8k":
        loader = HRBenchLoader(split_name="hrbench_8k", **kwargs)
    else:
        raise ValueError(
            f"Unknown dataset loader: {name}. Must be 'mme', 'vstar', "
            "'cvbench', 'hrbench-4k', or 'hrbench-8k'."
        )

    logger.info("dataloader_instantiated", dataset=name)
    return loader


__all__ = ["get_dataset_loader", "BaseDatasetLoader"]
