import importlib
from collections.abc import Awaitable, Callable

import structlog
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.types import Command

logger = structlog.get_logger(__name__)

#################################
### Base class for registries ###
#################################


class Registry[P]:  # P stands for Product
    def __init__(self) -> None:
        self._registry: dict[str, Callable[..., P]] = {}
        self._lazy_registry: dict[str, str] = {}

    def register(self, name: str) -> Callable[[Callable[..., P]], Callable[..., P]]:
        def decorator(func: Callable[..., P]) -> Callable[..., P]:
            if name in self._registry:
                raise ValueError(f"{name} already registered.")

            self._registry[name] = func
            return func

        return decorator

    def register_lazy(self, name: str, module_path: str) -> None:
        """Register a module path to be lazily imported when the item is requested."""
        self._lazy_registry[name] = module_path

    def get(self, name: str, **kwargs) -> P:
        if name not in self._registry and name in self._lazy_registry:
            try:
                # Import the module to trigger the @register decorator
                importlib.import_module(self._lazy_registry[name])
            except ImportError as e:
                raise ImportError(
                    f"Failed to lazy load module '{self._lazy_registry[name]}' for item '{name}'. "
                    f"Ensure optional dependencies are installed."
                ) from e

        if name not in self._registry:
            raise ValueError(f"{name} not found in {self.__class__.__name__}.")

        logger.debug("Retrieving from registry", registry=self.__class__.__name__, item=name)
        factory_fn = self._registry[name]
        return factory_fn(**kwargs)


#####################
### Tool Registry ###
#####################


# Inherit `Registry[P]' to make `self.__class__.__name__' unique for each registry
class ToolRegistry(Registry[StructuredTool]): ...


tool_registry = ToolRegistry()


###########################
### Chat Model Registry ###
###########################


class ChatModelRegistry(Registry[BaseChatModel]): ...


chat_model_registry = ChatModelRegistry()


#####################
### Node Registry ###
#####################

# A node is a function that takes the state and returns a dictionary of updates.
Node = (
    Callable[[dict], Awaitable[dict | Command]]
    | Callable[[dict, RunnableConfig], Awaitable[dict | Command]]
)


class NodeRegistry(Registry[Node]): ...


node_registry = NodeRegistry()


#################################
### Routing Function Registry ###
#################################

# A routing function is the argument `path' to be passed to `add_conditional_edges(source, path,
# path_map)'.
RoutingFunction = Callable[[dict], str]


class RoutingFunctionRegistry(Registry[RoutingFunction]): ...


routing_function_registry = RoutingFunctionRegistry()


#################################
### Prompt Generator Registry ###
#################################

# A prompt generator takes the graph state as input and outputs a prompt.  For now we only consider
# text prompts, and leave the processing of other types to the node itself.
PromptGenerator = Callable[[dict], str]


class PromptGeneratorRegistry(Registry[PromptGenerator]): ...


prompt_generator_registry = PromptGeneratorRegistry()


#################################
### Callback Handler Registry ###
#################################


class CallbackHandlerRegistry(Registry[BaseCallbackHandler]): ...


callback_handler_registry = CallbackHandlerRegistry()
