"""https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py"""

import math

from PIL import Image

from cv_agent.utils.storage import get_base64_url

MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768


def round_by_factor(number: int | float, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int | float, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by
    'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int | float, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,  # inferred from the repo for Qwen3-VL
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor**2)
    min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor**2)
    assert max_pixels >= min_pixels, (
        "The max_pixels of image must be greater than or equal to min_pixels."
    )
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        white_background = Image.new("RGB", image.size, (255, 255, 255))
        white_background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return image.convert("RGB")


def convert_for_qwen_vl(image: Image.Image) -> Image.Image:
    resized_height, resized_width = smart_resize(image.height, image.width)
    return to_rgb(image.resize((resized_width, resized_height)))


def get_base64_url_for_qwen_vl(image: Image.Image) -> str:
    return get_base64_url(convert_for_qwen_vl(image))


def rel_coords_to_abs(coords: list[int], real_resolution: tuple[int, int]) -> list[int]:
    lx, uy, rx, by = coords
    width, height = real_resolution
    scale_x = width / 1000
    scale_y = height / 1000
    return list(map(round, [lx * scale_x, uy * scale_y, rx * scale_x, by * scale_y]))


def abs_coords_to_rel(coords: list[int], real_resolution: tuple[int, int]) -> list[int]:
    lx, uy, rx, by = coords
    width, height = real_resolution
    scale_x = 1000 / width
    scale_y = 1000 / height
    return list(map(round, [lx * scale_x, uy * scale_y, rx * scale_x, by * scale_y]))
