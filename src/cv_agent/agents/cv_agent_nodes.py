"""Nodes for customized CV Agent."""

import asyncio
import copy
import json
import re
import time
from typing import Any

import structlog
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools.structured import StructuredTool

from cv_agent.constants import DEFAULT_TIMEOUT
from cv_agent.core.registries import (
    Node,
    RoutingFunction,
    node_registry,
    prompt_generator_registry,
    routing_function_registry,
)
from cv_agent.utils.storage import download_image_to_pil

logger = structlog.get_logger(__name__)


VISION_TOOL_NAMES = {
    "ocr",
    "detection",
    "detection_small_object",
    "segmentation",
    "cropping",
    "depth_estimation",
}

COORDINATE_INPUT_TOOLS = {
    "cropping",
}


def _extract_timing_metadata(response, base_log_data: dict) -> dict:
    """Extract rate limiter timing from response metadata if available."""
    if hasattr(response, "response_metadata") and isinstance(response.response_metadata, dict):
        wait_time = response.response_metadata.get("rate_limit_wait_seconds")
        api_time = response.response_metadata.get("api_request_seconds")

        if wait_time is not None and api_time is not None:
            base_log_data["latency_wait_seconds"] = wait_time
            base_log_data["latency_api_seconds"] = api_time

    return base_log_data


def generate_fake_url(state: dict, tool_name: str) -> str:
    """
    Generates a new, unique 'fake' URL for a generated image,
    based on the state's prefix and the tool that created it.
    """
    # 1. Get the prefix from the state (e.g., "cvbench_1")
    # This must be set in the initial state of your main script.
    prefix = state["prefix"]

    # 2. Create the base name for the new URL
    base_name = f"{prefix}_{tool_name}"  # e.g., "cvbench_1_cropping"

    key_prefix_to_check = f"http://{base_name}_"
    # 3. Count how many files with this base name already exist in the map
    url_map = state.get("url_map", {})
    count = 0
    for key in url_map:
        if key.startswith(key_prefix_to_check):
            count += 1

    # 4. The new number is the next one in the sequence
    new_number = count + 1

    # 5. Return the full fake URL
    return f"http://{base_name}_{new_number}.jpg"  # e.g., "cvbench_1_cropping_1.jpg"


@node_registry.register("planner_agent_node")
def get_planner_agent_node(
    chat_model: BaseChatModel,
    provided_tools: list[StructuredTool],
    prompt_generator: dict[str, Any],
    **kwargs,  # to be ignored
) -> Node:
    del kwargs
    generate_prompt = prompt_generator_registry.get(
        prompt_generator["name"], **prompt_generator.get("parameters", {})
    )
    chat_model_with_tools = chat_model.bind_tools(provided_tools)

    async def agent_node(state: dict) -> dict:
        messages = state.get("messages", [])

        current_turn = state["current_turn"]
        max_turns = state["max_turns"]
        if not messages:  # first round, need to initialize messages with question
            system_prompt = generate_prompt(state)
            messages.append(SystemMessage(content=system_prompt))

            original_image_url = state["original_figure_url"]
            user_question = state["question"]

            human_content = [
                {"type": "image_url", "image_url": {"url": original_image_url}},
                {"type": "text", "text": user_question},
                {"type": "text", "text": f"\nOrignal image URL: {original_image_url}\n"},
                {"type": "text", "text": f"\nYou are on TURN {current_turn} (max {max_turns})"},
            ]

            # Append the single, multi-modal HumanMessage
            messages.append(HumanMessage(content=human_content))
            logger.debug(
                "messages_initialized",
                message_count=len(messages),
                current_turn=current_turn,
                max_turns=max_turns,
            )

        else:
            reminder_text = (
                f"SYSTEM STATUS UPDATE: "
                f"You are currently on Turn {current_turn} out of {max_turns}.\n"
                f"If {current_turn} == {max_turns}, "
                f"you MUST stop calling tools and "
                f"provide your best final answer immediately in the required format."
            )
            messages.append(HumanMessage(content=reminder_text))

        logger.info(
            "llm_call_starting",
            current_turn=current_turn,
            max_turns=max_turns,
            message_count=len(messages),
        )

        start_time_total = time.perf_counter()
        response = await chat_model_with_tools.ainvoke(messages)
        latency_total = time.perf_counter() - start_time_total

        log_data = {
            "latency_total_seconds": round(latency_total, 2),
            "has_tool_calls": bool(response.tool_calls),
            "tool_count": len(response.tool_calls) if response.tool_calls else 0,
            "response_preview": str(response.content or "")[:200],
        }
        log_data = _extract_timing_metadata(response, log_data)
        logger.info("llm_call_completed", **log_data)

        # manually update the state
        messages.append(response)
        state.update({"messages": messages})
        # current turn increment
        state["current_turn"] = current_turn + 1

        return state

    return agent_node


async def _get_image_dimensions(state: dict, image_url: str) -> tuple[int, int]:
    """
    Gets image dimensions, reading from state first and downloading as a fallback.
    Saves new dimensions back to the state.
    """
    # 1. Check if dimensions are already in the state
    if "image_dimensions" not in state:
        state["image_dimensions"] = {}

    if image_url in state["image_dimensions"]:
        width, height = state["image_dimensions"][image_url]
        logger.debug(
            "image_dimensions_cached",
            image_url_preview=image_url[:50],
            width=width,
            height=height,
        )
        return width, height

    # 2. If not, download the image to get them
    logger.debug("image_dimensions_downloading", image_url_preview=image_url[:50])
    try:
        pil_image = await download_image_to_pil(image_url)
        width, height = pil_image.size

        # 3. Save new dimensions back to the state for future use
        state["image_dimensions"][image_url] = (width, height)

        return width, height
    except Exception as e:
        logger.exception(
            "image_dimensions_download_failed",
            image_url=image_url,
            error=str(e),
        )
        raise ValueError(f"Could not determine dimensions for image: {image_url}") from e


async def _convert_to_absolute_coords(
    state: dict, tool_args: dict, image_url: str, tool_name: str
) -> dict:
    """
    INBOUND: Converts relative [0, 1000] coords from Agent
    to absolute pixel coords for the Tool.
    """
    try:
        img_width, img_height = await _get_image_dimensions(state, image_url)
        if img_width == 0 or img_height == 0:
            raise ValueError(f"Invalid dimensions for image: {image_url}")

        scale_x = img_width / 1000.0
        scale_y = img_height / 1000.0

        if tool_name == "cropping" and "coordinates" in tool_args:
            rel_coords = tool_args["coordinates"]
            if len(rel_coords) == 4:
                tool_args["coordinates"] = [
                    int(rel_coords[0] * scale_x),
                    int(rel_coords[1] * scale_y),
                    int(rel_coords[2] * scale_x),
                    int(rel_coords[3] * scale_y),
                ]

        logger.debug(
            "coordinates_converted",
            direction="inbound",
            tool_name=tool_name,
            image_url_preview=image_url[:50],
        )
        return tool_args
    except Exception as e:
        logger.exception(
            "coordinate_conversion_failed",
            direction="inbound",
            tool_name=tool_name,
            error=str(e),
        )
        raise


def _scale_bbox(bbox: list[int | float], scale_x: float, scale_y: float) -> list[int]:
    return [
        int(bbox[0] * scale_x),
        int(bbox[1] * scale_y),
        int(bbox[2] * scale_x),
        int(bbox[3] * scale_y),
    ]


def _format_detection_text(detection: dict, bbox: list[int]) -> str:
    phrase = detection.get("phrase", "object")
    confidence = detection.get("confidence", 0.0)
    return (
        f"Detected {phrase} with confidence {confidence:.3f} "
        f"at location [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
    )


async def _convert_to_relative_coords(state: dict, tool_output_data: dict, image_url: str) -> dict:
    """
    OUTBOUND: Converts absolute pixel coords from detection tools
    to relative [0, 1000] coords for the agent.
    """
    try:
        img_width, img_height = await _get_image_dimensions(state, image_url)
        if img_width == 0 or img_height == 0:
            raise ValueError(f"Invalid dimensions for image: {image_url}")

        scale_x = 1000.0 / img_width
        scale_y = 1000.0 / img_height
        output_text = []

        for detection in tool_output_data.get("detections", []):
            bbox = detection.get("bbox")
            if bbox and len(bbox) == 4:
                relative_bbox = _scale_bbox(bbox, scale_x, scale_y)
                detection["bbox"] = relative_bbox
                output_text.append(_format_detection_text(detection, relative_bbox))

        if output_text:
            tool_output_data["output_text"] = output_text

        logger.debug(
            "coordinates_converted",
            direction="outbound",
            tool_name="detection",
            image_url_preview=image_url[:50],
        )
        return tool_output_data
    except Exception as e:
        logger.exception(
            "coordinate_conversion_failed",
            direction="outbound",
            tool_name="detection",
            error=str(e),
        )
        raise


@node_registry.register("cv_tool_executor")
def get_cv_tool_executor(executable_tools: dict[str, StructuredTool], **kwargs) -> Node:
    """
    This is the node that executes the tools.
    It acts as a two-way translation layer for coordinates.
    """
    del kwargs

    async def tool_executor(state: dict) -> dict:
        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return state

        tool_msgs = []

        tool_usage = state["tool_usage"]

        url_map = state["url_map"]

        for tool_call in last_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = copy.deepcopy(tool_call["args"])
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

            human_msg_to_add = None
            start_time = time.perf_counter()

            try:
                try:
                    tool = executable_tools[tool_name]
                except KeyError as e:
                    raise ValueError(f"No tool named '{tool_call['name']}'") from e

                # 1. Find the image URL this tool is supposed to act on
                # If the agent doesn't provide the 'image_url' argument,
                # it automatically defaults to "original_figure_url".
                image_key_to_check = tool_args.get("image_url", "original_figure_url")

                # 2. Check if this key exists in the map.
                # The 'url_map' contains "original_figure_url" AND all fake URLs.

                if image_key_to_check in url_map:
                    # It's a valid key. Use the real URL.
                    image_url_to_use = url_map[image_key_to_check]

                else:
                    # 3. This is the 'else' block you wanted.
                    # The key is not 'original_figure_url' and not a valid fake URL.
                    # We raise an error to send back to the agent.
                    raise ValueError(
                        f"The image key '{image_key_to_check}' does not exist in the url_map. "
                        f"Available keys are: {list(url_map.keys())}"
                    )

                # Replace to real url direct to the server
                tool_args["image_url"] = image_url_to_use
                # 2. INBOUND Translation: (Agent -> Tool)
                if tool_name in COORDINATE_INPUT_TOOLS:
                    tool_args = await _convert_to_absolute_coords(
                        state, tool_args, image_url_to_use, tool_name
                    )

                # 3. Run the tool
                from cv_agent.utils.concurrency import get_api_limiter

                async with get_api_limiter():
                    observation = await asyncio.wait_for(
                        tool.ainvoke(tool_args), timeout=DEFAULT_TIMEOUT
                    )

                latency = time.perf_counter() - start_time

                # 4. Parse the CallToolResult object (which might be string or dict)
                if not observation.content or observation.content[0].type != "text":
                    raise Exception("Tool returned invalid or non-text content")

                tool_output_content = observation.content[0].text

                if isinstance(tool_output_content, dict):
                    # It's already a dictionary, just use it
                    data = tool_output_content
                elif isinstance(tool_output_content, str):
                    # It's a string, so we need to parse it
                    data = json.loads(tool_output_content)
                else:
                    raise TypeError(f"Unknown tool output type: {type(tool_output_content)}")

                if "detection" in tool_name and data.get("status") == "success":
                    data = await _convert_to_relative_coords(state, data, image_url_to_use)

                real_image_url = data.get("output_image")

                if real_image_url and data.get("status") == "success":
                    fake_url = generate_fake_url(state, tool_name)
                    state["url_map"][fake_url] = real_image_url

                    # Append the new HumanMessage right after the ToolMessage
                    human_msg_to_add = HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": f"The tool successfully generated a new image. "
                                f"You can now refer to it as url: {fake_url}. It can"
                                f"works as tool input using the url.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": real_image_url},
                            },
                        ],
                    )

                final_tool_output = _post_process_tool_output(tool_name, data)

                # Canonical log line for successful tool execution
                logger.info(
                    "tool_executed",
                    tool_name=tool_name,
                    status="success",
                    latency_seconds=round(latency, 2),
                    input_args=str(tool_args)[:200],
                    output_preview=str(final_tool_output)[:200],
                    generated_image=bool(real_image_url),
                )

            except Exception as e:
                latency = time.perf_counter() - start_time if "start_time" in locals() else 0.0
                logger.exception(
                    "tool_execution_failed",
                    tool_name=tool_name,
                    latency_seconds=round(latency, 2),
                    input_args=str(tool_args)[:200],
                )
                final_tool_output = json.dumps(
                    {"status": "error", "message": f"Tool call failed: {e}"}
                )

            tool_msgs.append(ToolMessage(content=final_tool_output, tool_call_id=tool_call["id"]))

            if human_msg_to_add is not None:
                tool_msgs.append(human_msg_to_add)

        # 8. Update state
        state["messages"].extend(tool_msgs)
        state["tool_usage"] = tool_usage
        return state

    return tool_executor


@routing_function_registry.register("cv_agent_react_should_continue")
def get_cv_react_should_continue(
    end_dst: str = "end",
    continue_dst: str = "continue",
    use_hard_turn_limit: bool = True,
) -> RoutingFunction:
    def should_continue(state: dict) -> str:
        last_msg = state["messages"][-1]
        current_turn = state["current_turn"]
        max_turns = state["max_turns"]

        has_tool_calls = isinstance(last_msg, AIMessage) and bool(last_msg.tool_calls)

        if use_hard_turn_limit and current_turn > (max_turns + 2):
            decision = end_dst
            reason = "hard_limit"
        elif not has_tool_calls:
            decision = end_dst
            reason = "no_tool_calls"
        else:
            decision = continue_dst
            reason = "continue"

        logger.debug(
            "routing_decision",
            current_turn=current_turn,
            max_turns=max_turns,
            has_tool_calls=has_tool_calls,
            decision=decision,
            reason=reason,
        )

        return decision

    return should_continue


def _post_process_tool_output(tool_name: str, data: dict) -> str:
    """
    Cleans the raw tool output JSON, removing sensitive info (like real URLs)
    and simplifying the structure before sending it to the agent.
    """
    if data.get("status") != "success":
        # If the tool itself reported an error, just forward that.
        return json.dumps(
            {
                "status": "error",
                "message": data.get("message", "Tool reported an unspecified error."),
            }
        )

    # --- Tool-Specific Success Responses ---

    if "detection" in tool_name:
        results = data.get("output_text", data.get("detections", []))
        return json.dumps({"status": "success", "results": results})

    if "ocr" in tool_name:
        try:
            content = data.get("content", {})
            if not isinstance(content, dict) or not content:
                raise ValueError("OCR content is empty or not a dictionary.")

            first_item = next(iter(content.values()))
            text = first_item.get("md_content", "")

            if text is None:
                text_is_empty = True
            elif isinstance(text, str):
                stripped_text = text.strip()
                text_is_empty = not stripped_text or bool(
                    re.match(r"^\!\[.*?\]\(.*?\)$", stripped_text)
                )
            else:
                text_is_empty = False

            if text_is_empty:
                return json.dumps(
                    {"status": "success", "extracted_text": "No text detected in this image."}
                )

            return json.dumps({"status": "success", "extracted_text": text})
        except (AttributeError, StopIteration, ValueError) as e:
            logger.error("ocr_parse_failed", tool=tool_name, error=str(e))
            return json.dumps(
                {
                    "status": "error",
                    "message": "Failed to parse OCR output structure.",
                }
            )

    if tool_name == "cropping":
        # For cropping, include coordinate information if available
        result = {"status": "success", "message": "Image cropped successfully."}

        # Include original and modified coordinates if they were injected by BED
        if "original_coordinates" in data:
            result["original_coordinates"] = data["original_coordinates"]
        if "mcmc_selected_coordinates" in data:
            result["mcmc_selected_coordinates"] = data["mcmc_selected_coordinates"]
        if "look_ahead_selected_coordinates" in data:
            result["look_ahead_selected_coordinates"] = data["look_ahead_selected_coordinates"]
        if "boed_selected_coordinates" in data:
            result["boed_selected_coordinates"] = data["boed_selected_coordinates"]

        return json.dumps(result)

    else:
        return json.dumps(
            {"status": "success", "message": f"Tool '{tool_name}' executed successfully."}
        )
