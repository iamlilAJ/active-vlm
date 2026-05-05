from langfuse.langchain import CallbackHandler

from cv_agent.core.registries import callback_handler_registry


@callback_handler_registry.register("langfuse")
def get_langfuse_callback_handler() -> CallbackHandler:
    return CallbackHandler()
