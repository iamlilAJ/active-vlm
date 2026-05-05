"""LLMs hosted by OpenAI-compatible servers."""

import asyncio
import time

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from cv_agent.core.registries import chat_model_registry

logger = structlog.get_logger(__name__)


@chat_model_registry.register("openai_model")
def get_openai_model(
    model: str,
    base_url: str | None = None,
    max_retries: int = 3,
    **kwargs,
) -> ChatOpenAI:
    return ChatOpenAI(model=model, base_url=base_url, max_retries=max_retries, **kwargs)


class RateLimitedChatModel:
    """Wrapper that rate-limits all LLM invocations."""

    def __init__(self, model: BaseChatModel, limiter: asyncio.Semaphore):
        self._model = model
        self._limiter = limiter

    async def ainvoke(self, *args, **kwargs):
        # Measure time waiting for rate limiter
        wait_start = time.perf_counter()
        async with self._limiter:
            wait_time = time.perf_counter() - wait_start

            # Measure time for actual API call
            api_start = time.perf_counter()
            try:
                response = await self._model.ainvoke(*args, **kwargs)
                api_time = time.perf_counter() - api_start
            except Exception as e:
                api_time = time.perf_counter() - api_start
                logger.error(
                    "llm_api_call_failed",
                    wait_seconds=wait_time,
                    api_seconds=api_time,
                    error=str(e)[:200],
                )
                raise

        # Safely inject timing metadata
        try:
            if not hasattr(response, "response_metadata") or not isinstance(
                response.response_metadata, dict
            ):
                response.response_metadata = {}
            response.response_metadata["rate_limit_wait_seconds"] = wait_time
            response.response_metadata["api_request_seconds"] = api_time
        except (AttributeError, TypeError) as e:
            logger.warning(
                "timing_metadata_injection_failed",
                error=str(e),
                wait_seconds=wait_time,
                api_seconds=api_time,
            )

        return response

    def __getattr__(self, name):
        # Delegate all other methods to wrapped model
        return getattr(self._model, name)


@chat_model_registry.register("openai_rate_limited")
def get_rate_limited_openai_model(
    model: str,
    base_url: str | None = None,
    **kwargs,
) -> RateLimitedChatModel:
    """Create an OpenAI chat model with global rate limiting.

    Args:
        model: OpenAI model name
        base_url: Optional base URL for OpenAI-compatible servers
        **kwargs: Additional arguments passed to ChatOpenAI

    Returns:
        Rate-limited chat model wrapper
    """
    from cv_agent.utils.concurrency import get_api_limiter

    raw_model = get_openai_model(model=model, base_url=base_url, **kwargs)
    return RateLimitedChatModel(raw_model, get_api_limiter())
