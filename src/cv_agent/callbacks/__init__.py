from typing import TYPE_CHECKING

from ..core.registries import callback_handler_registry

if TYPE_CHECKING:
    from .langfuse import get_langfuse_callback_handler

# Lazy register for configuration-based access
callback_handler_registry.register_lazy("langfuse", "cv_agent.callbacks.langfuse")

__all__ = ["get_langfuse_callback_handler"]


def __getattr__(name: str):
    if name == "get_langfuse_callback_handler":
        try:
            from . import langfuse

            return langfuse.get_langfuse_callback_handler
        except ImportError as e:
            raise ImportError(
                f"Failed to import optional dependency for '{name}'. "
                "Please ensure 'langfuse' is installed."
            ) from e

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
