import logging
import sys
from pathlib import Path

import orjson
import structlog
from omegaconf import DictConfig

LEVEL_MAP = logging.getLevelNamesMapping()


def setup_logging(config: DictConfig | None = None) -> None:
    # global defaults
    default_level = "INFO"
    default_json = False
    handlers_config = []

    if config is not None:
        default_level = config.get("level", default_level).upper()
        default_json = config.get("json", default_json)
        handlers_config = config.get("handlers", [])

    if not handlers_config:
        handlers_config = [{"type": "console", "level": default_level, "json": default_json}]

    shared_processors = get_shared_processors()

    handlers = []
    min_level = logging.CRITICAL

    for cfg in handlers_config:
        handler, min_level = make_handler(
            cfg, shared_processors, default_level, default_json, min_level
        )
        handlers.append(handler)

    # front-end
    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.make_filtering_bound_logger(min_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # the legacy back-end
    root_logger = logging.getLogger()
    root_logger.handlers = handlers
    root_logger.setLevel(min_level)


def get_shared_processors() -> list:
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]


def make_handler(
    cfg: DictConfig | dict,
    shared_processors: list,
    default_level: str,
    default_json: bool,
    min_level: int,
) -> tuple[logging.Handler, int]:
    htype = cfg.get("type", "console")
    hlevel = cfg.get("level", default_level).upper()
    hjson = cfg.get("json", default_json)

    try:
        level_val = LEVEL_MAP[hlevel]
    except Exception:
        level_val = logging.INFO

    if level_val < min_level:
        min_level = level_val

    if hjson:
        renderer = structlog.processors.JSONRenderer(
            serializer=lambda *args, **kwargs: orjson.dumps(*args, **kwargs).decode()
        )
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=htype == "console")

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )

    if htype == "console":
        handler = logging.StreamHandler(sys.stdout)
    elif htype == "file":
        path_str = cfg.get("path")
        if path_str is not None:
            path = Path(path_str).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(path)
        else:
            raise ValueError(f"For a file handler, a path must be provided. ({repr(cfg)})")
    else:
        raise NotImplementedError(f"Handler of type {repr(htype)} not implemented yet.")

    handler.setFormatter(formatter)
    handler.setLevel(level_val)

    return handler, min_level
