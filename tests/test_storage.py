import pytest
from PIL import Image

from cv_agent.utils.storage import get_base64_url, get_minio_config, image_to_jpeg_bytes, parse_bool


def test_get_minio_config_from_explicit_env() -> None:
    config = get_minio_config(
        {
            "CV_AGENT_MINIO_ENDPOINT": "localhost:9000",
            "CV_AGENT_MINIO_ACCESS_KEY": "access",
            "CV_AGENT_MINIO_SECRET_KEY": "secret",
            "CV_AGENT_MINIO_BUCKET": "bucket",
            "CV_AGENT_MINIO_SECURE": "true",
        }
    )

    assert config.endpoint == "localhost:9000"
    assert config.access_key == "access"
    assert config.secret_key == "secret"
    assert config.bucket_name == "bucket"
    assert config.secure is True


def test_get_minio_config_requires_credentials() -> None:
    with pytest.raises(RuntimeError, match="CV_AGENT_MINIO_ENDPOINT"):
        get_minio_config({})


def test_parse_bool_defaults_and_truthy_values() -> None:
    assert parse_bool(None, default=True) is True
    assert parse_bool("false") is False
    assert parse_bool("1") is True
    assert parse_bool("yes") is True


def test_image_encoding_helpers_are_deterministic_shape() -> None:
    image = Image.new("RGBA", (8, 6), (255, 0, 0, 128))

    encoded = image_to_jpeg_bytes(image)
    url = get_base64_url(image)

    assert encoded.startswith(b"\xff\xd8")
    assert url.startswith("data:image/jpeg;base64,")
