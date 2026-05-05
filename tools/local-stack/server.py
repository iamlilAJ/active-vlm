"""Local FastMCP server exposing the cropping tool used by active-vlm.

Runs on the host (not in docker) so the MinIO URL it returns is reachable from
both the agent process and any downstream tool/VLM. Configuration is via env
vars; defaults match the bundled ``docker-compose.yml``.
"""

import os
import time
import uuid
from io import BytesIO


def _disable_proxy_for_localhost() -> None:
    """Make sure HTTP_PROXY, if set, does not intercept localhost traffic.

    httpx and the minio client both honor ``NO_PROXY`` at construction time;
    appending ``localhost,127.0.0.1`` is a defensive no-op when no proxy is set.
    """
    no_proxy = os.environ.get("NO_PROXY", "")
    if "localhost" not in no_proxy:
        os.environ["NO_PROXY"] = (no_proxy + ",localhost,127.0.0.1").lstrip(",")
    os.environ["no_proxy"] = os.environ["NO_PROXY"]


_disable_proxy_for_localhost()

import httpx  # noqa: E402
import structlog  # noqa: E402
from _minio import get_minio_config, make_minio_client, public_url_for  # noqa: E402
from fastmcp import FastMCP  # noqa: E402
from PIL import Image  # noqa: E402

logger = structlog.get_logger(__name__)

MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))
DOWNLOAD_TIMEOUT = float(os.environ.get("CROP_DOWNLOAD_TIMEOUT", "30"))

CONFIG = get_minio_config()
MINIO_CLIENT = make_minio_client(CONFIG)
mcp = FastMCP("cropping-local")


def image_to_jpeg_bytes(image: Image.Image, quality: int = 90) -> bytes:
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


@mcp.tool(
    name="crop_image_tool_crop_image_post",
    description="Crop an image at absolute pixel coordinates [x1, y1, x2, y2] "
    "and return the public URL of the resulting JPEG.",
)
async def crop_image_tool_crop_image_post(
    image_url: str, x1: int, y1: int, x2: int, y2: int
) -> dict:
    """Crop the given image and upload the result to MinIO.

    Returns a dict matching the contract in ``cv_agent_nodes.py:393-397``::

        {"status": "success", "output_image": "<public url>"}
    """
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT) as client:
        response = await client.get(image_url)
        response.raise_for_status()

    image = Image.open(BytesIO(response.content))
    cropped = image.crop((int(x1), int(y1), int(x2), int(y2)))
    body = image_to_jpeg_bytes(cropped)

    object_name = f"crops/{uuid.uuid4().hex}.jpg"
    MINIO_CLIENT.put_object(
        CONFIG.bucket_name,
        object_name,
        BytesIO(body),
        length=len(body),
        content_type="image/jpeg",
    )

    public_url = public_url_for(CONFIG, object_name)
    logger.info(
        "crop_uploaded",
        source_url=image_url,
        coordinates=[int(x1), int(y1), int(x2), int(y2)],
        cropped_size=cropped.size,
        public_url=public_url,
        latency_seconds=round(time.perf_counter() - started, 3),
    )
    return {"status": "success", "output_image": public_url}


if __name__ == "__main__":
    logger.info(
        "server_starting",
        host=MCP_HOST,
        port=MCP_PORT,
        bucket=CONFIG.bucket_name,
        endpoint=CONFIG.endpoint,
        public_base_url=CONFIG.public_base_url,
    )
    mcp.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT, path="/mcp")
