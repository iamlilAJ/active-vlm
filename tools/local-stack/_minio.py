"""Shared MinIO config + client helpers for the local stack.

Mirrors the style of ``cv_agent.utils.storage`` but stays self-contained — the
local stack has its own venv and does not depend on ``cv_agent``.
"""

import os
from dataclasses import dataclass

from minio import Minio


@dataclass(frozen=True)
class MinioConfig:
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    public_base_url: str
    secure: bool


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_minio_config() -> MinioConfig:
    endpoint = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
    return MinioConfig(
        endpoint=endpoint,
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        bucket_name=os.environ.get("MINIO_BUCKET", "cv-agent"),
        public_base_url=os.environ.get("MINIO_PUBLIC_BASE_URL", endpoint),
        secure=parse_bool(os.environ.get("MINIO_SECURE")),
    )


def make_minio_client(config: MinioConfig) -> Minio:
    netloc = config.endpoint.removeprefix("https://").removeprefix("http://").rstrip("/")
    return Minio(
        netloc,
        access_key=config.access_key,
        secret_key=config.secret_key,
        secure=config.secure,
    )


def public_url_for(config: MinioConfig, object_name: str) -> str:
    return f"{config.public_base_url.rstrip('/')}/{config.bucket_name}/{object_name}"
