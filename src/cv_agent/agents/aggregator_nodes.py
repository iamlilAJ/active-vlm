import time
from typing import Any

import structlog
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from cv_agent.core.registries import (
    Node,
    node_registry,
    prompt_generator_registry,
)

logger = structlog.get_logger(__name__)


def _extract_timing_metadata(response, base_log_data: dict) -> dict:
    """Extract rate limiter timing from response metadata if available."""
    if hasattr(response, "response_metadata") and isinstance(response.response_metadata, dict):
        wait_time = response.response_metadata.get("rate_limit_wait_seconds")
        api_time = response.response_metadata.get("api_request_seconds")

        if wait_time is not None and api_time is not None:
            base_log_data["latency_wait_seconds"] = wait_time
            base_log_data["latency_api_seconds"] = api_time

    return base_log_data


@node_registry.register("direct_reasoning_node")
def get_direct_reasoning_node(
    chat_model: BaseChatModel,
    prompt_generator: dict[str, Any],
    **kwargs,  # to be ignored
) -> Node:
    """
    Creates a node that performs direct, tool-less reasoning on the
    original question and image.
    """
    del kwargs
    generate_prompt = prompt_generator_registry.get(
        prompt_generator["name"], **prompt_generator.get("parameters", {})
    )

    async def direct_reasoning_node(state: dict) -> dict:
        # 1. Generate the system prompt (which includes the question)
        system_prompt = generate_prompt(state)
        original_image_url = state["original_figure_url"]

        # 2. Build the multi-modal message
        human_content = [
            {"type": "text", "text": system_prompt},
            {"type": "image_url", "image_url": {"url": original_image_url}},
        ]

        messages = [HumanMessage(content=human_content)]

        # 3. Call the model
        logger.info(
            "llm_call_starting",
            node_type="direct_reasoning",
            has_image=bool(original_image_url),
        )

        start_time = time.perf_counter()
        response = await chat_model.ainvoke(messages)
        latency = time.perf_counter() - start_time

        log_data = {
            "node_type": "direct_reasoning",
            "latency_total_seconds": round(latency, 2),
            "result_preview": str(response.content or "")[:200],
        }
        log_data = _extract_timing_metadata(response, log_data)
        logger.info("llm_call_completed", **log_data)

        # 4. Save the result to the new state field
        return {"direct_reasoning_result": response.content}

    return direct_reasoning_node


def _build_aggregator_messages(state: dict, system_prompt_str: str) -> list[BaseMessage]:
    """
    Implements the "structured report" design.
    Converts the ReAct history into a series of HumanMessages for the aggregator,
    one for each "turn".
    """

    messages_for_aggregator: list[BaseMessage] = []

    # Add the System Prompt (loaded from the .j2 file)
    messages_for_aggregator.append(SystemMessage(content=system_prompt_str))

    # Add the Original Question and Image (as the first "report")
    original_question = state.get("question", "No question provided.")
    original_image_url = state.get("original_figure_url")

    original_content: list[dict] = [
        {"type": "text", "text": f"Original User Question:\n{original_question}"}
    ]
    if original_image_url:
        original_content.append({"type": "image_url", "image_url": {"url": original_image_url}})

    messages_for_aggregator.append(HumanMessage(content=original_content))

    #  Add a separator
    messages_for_aggregator.append(
        HumanMessage(content="--- Turn-by-Turn Report from ReAct Agent (with tools) ---")
    )

    # 5. Process the ReAct agent's history
    react_messages = state.get("messages", [])
    if not react_messages:
        return messages_for_aggregator

    # Start iterating *after* the first System and Human message
    i = 0
    # Find the first AI message to start processing
    while i < len(react_messages) and not isinstance(react_messages[i], AIMessage):
        i += 1

    # Now, process all subsequent turns
    while i < len(react_messages):
        msg = react_messages[i]

        if isinstance(msg, AIMessage):
            # This is the start of a turn
            if not msg.tool_calls and msg.content:
                # --- This is a FINAL ANSWER from ReAct ---
                report_content = f"Finally, the ReAct agent's analysis is:\n\n{msg.content}"
                messages_for_aggregator.append(HumanMessage(content=report_content))
                i += 1

            elif msg.tool_calls:
                # --- This is a TOOL CALL turn ---
                ai_msg = msg

                if (i + 1) < len(react_messages) and isinstance(react_messages[i + 1], ToolMessage):
                    tool_msg = react_messages[i + 1]

                    if (i + 2) < len(react_messages) and isinstance(
                        react_messages[i + 2], HumanMessage
                    ):
                        # This is the (AI -> Tool -> Human) case
                        human_img_msg = react_messages[i + 2]

                        report_content = []
                        report_content.append(
                            {
                                "type": "text",
                                "text": f"AGENT THOUGHT:\n{ai_msg.content}\n\n"
                                f"TOOL OBSERVATION:\n{tool_msg.content}",
                            }
                        )

                        if isinstance(human_img_msg.content, list):
                            report_content.extend(human_img_msg.content)
                        else:
                            report_content.append({"type": "text", "text": human_img_msg.content})

                        messages_for_aggregator.append(HumanMessage(content=report_content))
                        i += 3  # Skip AI, Tool, and Human messages

                    else:
                        # This is the (AI -> Tool) case (no new image)
                        report_content = (
                            f"AGENT THOUGHT:\n{ai_msg.content}\n\n"
                            f"TOOL OBSERVATION:\n{tool_msg.content}"
                        )
                        messages_for_aggregator.append(HumanMessage(content=report_content))
                        i += 2  # Skip AI and Tool messages

                else:
                    logger.warning("ai_message_without_tool_message", message_index=i)
                    i += 1
            else:
                i += 1  # Empty AI message, just skip
        else:
            i += 1  # Stray message, skip to find next AI

    #  Add the Direct Reasoning (as the second "report")
    direct_reasoning_result = state.get("direct_reasoning_result", "No reasoning provided.")
    messages_for_aggregator.append(
        HumanMessage(
            content=f"Analysis from Direct Reasoning (no tools):\n\n{direct_reasoning_result}"
        )
    )

    #  Add the final, conclusive instruction
    messages_for_aggregator.append(
        HumanMessage(
            content="--- Final Task ---\nYou have now seen the original request, "
            "the full ReAct agent's report,"
            " and the Direct Reasoning report. "
            "Please synthesize all of this information and"
            " provide your final, definitive answer."
        )
    )

    return messages_for_aggregator


@node_registry.register("final_aggregator_node")
def get_final_aggregator_node(
    chat_model: BaseChatModel,
    prompt_generator: dict[str, Any],
    **kwargs,  # to be ignored
) -> Node:
    """
    Creates the final aggregator node.
    It loads its system prompt from the registry and uses the
    _build_aggregator_messages helper to construct its full message history.
    """
    del kwargs
    generate_prompt = prompt_generator_registry.get(
        prompt_generator["name"], **prompt_generator.get("parameters", {})
    )

    async def final_aggregator_node(state: dict) -> dict:
        # 1. Get the static system prompt string
        system_prompt_str = generate_prompt(state)

        # 2. Build the new message list using your "grouping" logic
        messages_for_aggregator = _build_aggregator_messages(state, system_prompt_str)

        # 3. Call the LLM with the structured report
        logger.info(
            "llm_call_starting",
            node_type="final_aggregator",
            message_count=len(messages_for_aggregator),
        )

        start_time = time.perf_counter()
        response = await chat_model.ainvoke(messages_for_aggregator)
        latency = time.perf_counter() - start_time

        log_data = {
            "node_type": "final_aggregator",
            "latency_total_seconds": round(latency, 2),
            "answer_preview": str(response.content or "")[:200],
        }
        log_data = _extract_timing_metadata(response, log_data)
        logger.info("llm_call_completed", **log_data)

        # 4. Return the new 'messages' list for the state.
        return {"messages": [response]}

    return final_aggregator_node
