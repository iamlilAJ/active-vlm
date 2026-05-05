"""Image storage and encoding utilities.

Runtime storage credentials are read from environment variables at the edge.
Core image conversion helpers stay deterministic and easy to test.
"""

import base64
import os
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO

import anyio
import backoff
import httpx
import structlog
from minio import Minio
from PIL import Image

from cv_agent.constants import DEFAULT_TIMEOUT

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class MinioConfig:
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    secure: bool


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _required_env(env: Mapping[str, str], name: str) -> str:
    value = env.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_minio_config(env: Mapping[str, str] | None = None) -> MinioConfig:
    source = os.environ if env is None else env
    return MinioConfig(
        endpoint=_required_env(source, "CV_AGENT_MINIO_ENDPOINT"),
        access_key=_required_env(source, "CV_AGENT_MINIO_ACCESS_KEY"),
        secret_key=_required_env(source, "CV_AGENT_MINIO_SECRET_KEY"),
        bucket_name=source.get("CV_AGENT_MINIO_BUCKET", "cv-agent"),
        secure=parse_bool(source.get("CV_AGENT_MINIO_SECURE"), default=False),
    )


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()


class MinioManager:
    def __init__(self, config: MinioConfig):
        self._config = config
        self._client = Minio(
            config.endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=config.secure,
        )

    def ensure_bucket(self) -> None:
        if self._client.bucket_exists(self._config.bucket_name):
            return
        self._client.make_bucket(self._config.bucket_name)

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def upload_file(self, object_name: str, file_path: str) -> str:
        self.ensure_bucket()
        self._client.fput_object(self._config.bucket_name, object_name, file_path)
        protocol = "https" if self._config.secure else "http"
        return f"{protocol}://{self._config.endpoint}/{self._config.bucket_name}/{object_name}"


def _upload_image_bytes(image_bytes: bytes, object_name: str, config: MinioConfig) -> str:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file.write(image_bytes)
        temp_file_path = tmp_file.name

    try:
        return MinioManager(config).upload_file(object_name, temp_file_path)
    finally:
        os.unlink(temp_file_path)


def upload_pil_image_to_minio(image: Image.Image, sample_id: str) -> tuple[str, int, int]:
    width, height = image.size
    object_name = f"cvagent/initial/{sample_id}.jpg"

    try:
        public_url = _upload_image_bytes(
            image_to_jpeg_bytes(image), object_name, get_minio_config()
        )
    except Exception:
        logger.exception("image_upload_failed", sample_id=sample_id)
        raise

    logger.info("image_uploaded", sample_id=sample_id, url=public_url, width=width, height=height)
    return public_url, width, height


def upload_new_artifact(image: Image.Image) -> tuple[str, int, int]:
    width, height = image.size
    object_name = f"cvagent/artifacts/{uuid.uuid4()}.jpg"

    try:
        public_url = _upload_image_bytes(
            image_to_jpeg_bytes(image), object_name, get_minio_config()
        )
    except Exception:
        logger.exception("artifact_upload_failed")
        return "", 0, 0

    logger.info("artifact_uploaded", url=public_url, width=width, height=height)
    return public_url, width, height


@backoff.on_exception(backoff.expo, httpx.RequestError, max_tries=3)
async def download_image_to_pil(image_url: str) -> Image.Image:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
    except Exception:
        logger.exception("image_download_failed", image_url=image_url)
        raise


def encode_pil_image(image: Image.Image) -> str:
    return base64.b64encode(image_to_jpeg_bytes(image)).decode("utf-8")


def get_base64_url(image: Image.Image) -> str:
    return f"data:image/jpeg;base64,{encode_pil_image(image)}"


async def aload_image(path: anyio.Path) -> Image.Image:
    image_raw = await path.read_bytes()
    return Image.open(BytesIO(image_raw))


async def asave_image(path: anyio.Path, image: Image.Image) -> None:
    await path.write_bytes(image_to_jpeg_bytes(image))
