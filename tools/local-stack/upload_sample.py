"""Synthesize a sample image and upload it to the local MinIO bucket.

Reads the same env vars as ``server.py``. Prints the public URL on stdout so
the caller can pipe it into ``smoke_test.py`` or feed it to the agent.
"""

import os
from io import BytesIO


def _disable_proxy_for_localhost() -> None:
    no_proxy = os.environ.get("NO_PROXY", "")
    if "localhost" not in no_proxy:
        os.environ["NO_PROXY"] = (no_proxy + ",localhost,127.0.0.1").lstrip(",")
    os.environ["no_proxy"] = os.environ["NO_PROXY"]


_disable_proxy_for_localhost()

from _minio import MinioConfig, get_minio_config, make_minio_client, public_url_for  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

DEFAULT_OBJECT_NAME = "samples/sample.jpg"


def make_sample_image(width: int = 640, height: int = 480) -> bytes:
    """Generate a 4-quadrant test image with distinct colors per region."""
    image = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(image)

    quadrants = [
        ((0, 0, width // 2, height // 2), (200, 70, 70)),
        ((width // 2, 0, width, height // 2), (70, 160, 200)),
        ((0, height // 2, width // 2, height), (90, 180, 90)),
        ((width // 2, height // 2, width, height), (220, 190, 70)),
    ]
    for box, color in quadrants:
        draw.rectangle(box, fill=color)

    for i in range(0, width, 80):
        draw.line([(i, 0), (i, height)], fill=(255, 255, 255), width=1)
    for j in range(0, height, 80):
        draw.line([(0, j), (width, j)], fill=(255, 255, 255), width=1)

    draw.text((20, 20), "active-vlm sample", fill=(0, 0, 0))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()


def upload_sample(config: MinioConfig, object_name: str = DEFAULT_OBJECT_NAME) -> str:
    body = make_sample_image()
    client = make_minio_client(config)
    client.put_object(
        config.bucket_name,
        object_name,
        BytesIO(body),
        length=len(body),
        content_type="image/jpeg",
    )
    return public_url_for(config, object_name)


def main() -> None:
    print(upload_sample(get_minio_config()))


if __name__ == "__main__":
    main()
