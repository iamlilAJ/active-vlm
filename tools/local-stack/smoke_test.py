"""End-to-end smoke test for the local cropping MCP server.

Exercises the same ``CroppingTool`` client the agent uses, against the local
server + MinIO. Run from the **main repo root** so ``cv_agent`` is importable::

    cd /path/to/active-vlm
    uv run python tools/local-stack/smoke_test.py

Prereqs: docker compose stack up, ``server.py`` running, sample image uploaded.
"""

import asyncio
import json
import os
import sys


def _disable_proxy_for_localhost() -> None:
    no_proxy = os.environ.get("NO_PROXY", "")
    if "localhost" not in no_proxy:
        os.environ["NO_PROXY"] = (no_proxy + ",localhost,127.0.0.1").lstrip(",")
    os.environ["no_proxy"] = os.environ["NO_PROXY"]


_disable_proxy_for_localhost()

import httpx  # noqa: E402

from cv_agent.tools.cv_toolkit import CroppingTool  # noqa: E402

MCP_URL = os.environ.get("CV_AGENT_CROPPING_MCP_URL", "http://localhost:8000/mcp")

_MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
_MINIO_PUBLIC_BASE_URL = os.environ.get("MINIO_PUBLIC_BASE_URL", _MINIO_ENDPOINT)
_MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "cv-agent")
SAMPLE_URL = os.environ.get(
    "SAMPLE_IMAGE_URL",
    f"{_MINIO_PUBLIC_BASE_URL.rstrip('/')}/{_MINIO_BUCKET}/samples/sample.jpg",
)
CROP_COORDINATES = [10, 10, 200, 200]


async def main() -> int:
    tool = CroppingTool(MCP_URL)
    result = await tool.invoke(image_url=SAMPLE_URL, coordinates=CROP_COORDINATES)

    if not result.content or result.content[0].type != "text":
        print(f"FAIL: unexpected MCP result content: {result.content!r}", file=sys.stderr)
        return 1

    data = json.loads(result.content[0].text)
    print(f"data={data}")

    if data.get("status") != "success":
        print(f"FAIL: status was {data.get('status')!r}", file=sys.stderr)
        return 1

    output_image = data.get("output_image")
    if not output_image:
        print("FAIL: missing output_image", file=sys.stderr)
        return 1

    response = httpx.get(output_image, timeout=10)
    print(f"GET {output_image} -> {response.status_code} {response.headers.get('content-type')}")
    if response.status_code != 200:
        print(f"FAIL: HTTP {response.status_code}", file=sys.stderr)
        return 1
    if not response.headers.get("content-type", "").startswith("image/"):
        print("FAIL: response is not an image", file=sys.stderr)
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
