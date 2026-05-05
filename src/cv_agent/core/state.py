from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def create_agent_state(state_definition: dict[str, str]):
    parsed_state_definition = {
        k: eval(
            v,
            {
                "str": str,
                "bool": bool,
                "list": list,
                "dict": dict,
                "Any": Any,
                "Annotated": Annotated,
                "AnyMessage": AnyMessage,
                "add_messages": add_messages,
            },  # safe eval by handcrafting parsable keywords
        )
        for k, v in state_definition.items()
    }
    return TypedDict("AgentGraph", parsed_state_definition)  # type: ignore
