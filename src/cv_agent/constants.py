import os
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

type Bbox = tuple[int, int, int, int]

try:
    DEFAULT_TIMEOUT = int(os.getenv("CV_AGENT_TIMEOUT", "90"))
except Exception as e:
    logger.exception(
        "constant_parse_error",
        error=f"{e}",
        envvar=f"CV_AGENT_TIMEOUT={os.getenv('CV_AGENT_TIMEOUT')}",
    )
    DEFAULT_TIMEOUT = 90

SRC_DIR = Path(__file__).parent
PROMPT_DIR = SRC_DIR / "prompts" / "templates"
