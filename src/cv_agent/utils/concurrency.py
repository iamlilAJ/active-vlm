import asyncio

# Global singleton
_API_LIMITER: asyncio.Semaphore | None = None


def initialize_api_limiter(limit: int) -> None:
    """Initialize the global API concurrency limiter.

    Args:
        limit: Maximum number of concurrent API requests
    """
    global _API_LIMITER
    _API_LIMITER = asyncio.Semaphore(limit)


def get_api_limiter() -> asyncio.Semaphore:
    """Get the global API limiter instance."""
    if _API_LIMITER is None:
        raise RuntimeError("API limiter not initialized. Call initialize_api_limiter() first.")
    return _API_LIMITER
