"""BED (Bayesian Experimental Design) node for CV Agent.

This module implements a tool executor that intercepts cropping calls and applies
Bayesian Experimental Design to select the optimal crop region.
"""

import asyncio
import copy
import json
import time
from pathlib import Path

import structlog
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools.structured import StructuredTool

from cv_agent.agents.cv_agent_nodes import (
    COORDINATE_INPUT_TOOLS,
    _convert_to_absolute_coords,
    _convert_to_relative_coords,
    _post_process_tool_output,
    generate_fake_url,
)
from cv_agent.constants import DEFAULT_TIMEOUT, PROMPT_DIR, Bbox
from cv_agent.core.registries import Node, node_registry
from cv_agent.policies.boed import BoedPolicy
from cv_agent.policies.look_ahead import LookAheadPolicy
from cv_agent.policies.mcmc import McmcPolicy
from cv_agent.utils.concurrency import get_api_limiter
from cv_agent.utils.storage import download_image_to_pil

logger = structlog.get_logger(__name__)


@node_registry.register("look_ahead_crop_tool_executor")
def get_look_ahead_crop_tool_executor(
    executable_tools: dict[str, StructuredTool],
    chat_model: BaseChatModel,
    inner_crop_prompt: Path | str = PROMPT_DIR / "look_ahead_crop_inst.md",
    resolvability_prompt: Path | str = PROMPT_DIR / "look_ahead_resolvability.md",
    scaling_factors: list[float] = [1.5, 1, 0.8],  # noqa: B006
    n_samples: int = 5,
    **kwargs,
) -> Node:
    del kwargs

    # Get system prompts
    inner_crop_sys_inst = Path(inner_crop_prompt).read_text()
    resolvability_sys_inst = Path(resolvability_prompt).read_text()

    # Setup LOOK_AHEAD
    look_ahead = LookAheadPolicy(scaling_factors, n_samples)

    async def look_ahead_tool_executor(state: dict) -> dict:
        """Execute tools with LOOK_AHEAD interception for cropping."""
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

                # 1. Find the image URL
                image_key_to_check = tool_args.get("image_url", state["original_figure_url"])

                if image_key_to_check in url_map:
                    image_url_to_use = url_map[image_key_to_check]
                else:
                    raise ValueError(
                        f"The image key '{image_key_to_check}' does not exist in the url_map. "
                        f"Available keys are: {list(url_map.keys())}"
                    )

                tool_args["image_url"] = image_url_to_use

                # 2. LOOK-AHEAD INTERCEPTION for cropping BEFORE coordinate conversion
                original_coords = None
                modified_coords = None
                if tool_name == "cropping":
                    original_coords = tool_args["coordinates"].copy()
                    validated_coords = validate_crop_coordinates(original_coords)
                    try:
                        original_image = await download_image_to_pil(state["original_figure_url"])
                        if image_url_to_use == state["original_figure_url"]:
                            image_to_crop = None
                        else:
                            image_to_crop = await download_image_to_pil(image_url_to_use)
                    except Exception:
                        logger.exception("look_ahead_image_download_failed")
                        modified_coords = tool_args["coordinates"].copy()
                    else:
                        bbox = await look_ahead.get_action(
                            chat_model,
                            state["question"],
                            original_image,
                            image_to_crop,
                            validated_coords,
                            inner_crop_sys_inst,
                            resolvability_sys_inst,
                        )
                        modified_coords = list(bbox)
                        tool_args["coordinates"] = modified_coords

                # 3. INBOUND coordinate conversion (Agent -> Tool)
                if tool_name in COORDINATE_INPUT_TOOLS:
                    tool_args = await _convert_to_absolute_coords(
                        state, tool_args, image_url_to_use, tool_name
                    )

                # 4. Run the tool
                async with get_api_limiter():
                    observation = await asyncio.wait_for(
                        tool.ainvoke(tool_args), timeout=DEFAULT_TIMEOUT
                    )

                latency = time.perf_counter() - start_time

                # 5. Parse the CallToolResult object
                if not observation.content or observation.content[0].type != "text":
                    raise Exception("Tool returned invalid or non-text content")

                tool_output_content = observation.content[0].text

                if isinstance(tool_output_content, dict):
                    data = tool_output_content
                elif isinstance(tool_output_content, str):
                    data = json.loads(tool_output_content)
                else:
                    raise TypeError(f"Unknown tool output type: {type(tool_output_content)}")

                if "detection" in tool_name and data.get("status") == "success":
                    data = await _convert_to_relative_coords(state, data, image_url_to_use)

                # 7. Inject LOOK_AHEAD-modified coordinates for cropping tool
                if tool_name == "cropping" and data.get("status") == "success":
                    if original_coords is not None and modified_coords is not None:
                        data["original_coordinates"] = original_coords
                        data["look_ahead_selected_coordinates"] = modified_coords

                real_image_url = data.get("output_image")

                if real_image_url and data.get("status") == "success":
                    fake_url = generate_fake_url(state, tool_name)
                    state["url_map"][fake_url] = real_image_url

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

                # Canonical log line
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

        # Update state
        state["messages"].extend(tool_msgs)
        state["tool_usage"] = tool_usage
        return state

    return look_ahead_tool_executor


@node_registry.register("mcmc_crop_tool_executor")
def get_mcmc_crop_tool_executor(
    executable_tools: dict[str, StructuredTool],
    chat_model: BaseChatModel,
    resolvability_prompt: Path | str = PROMPT_DIR / "look_ahead_resolvability.md",
    n_iterations: int = 6,
    n_samples: int = 3,
    eta: float = 0.15,
    acceptance_floor: float = 0.1,
    jitter_ratio: float = 0.05,
    **kwargs,
) -> Node:
    del kwargs

    resolvability_sys_inst = Path(resolvability_prompt).read_text()
    mcmc = McmcPolicy(
        n_iterations=n_iterations,
        n_samples=n_samples,
        eta=eta,
        acceptance_floor=acceptance_floor,
        jitter_ratio=jitter_ratio,
    )

    async def mcmc_tool_executor(state: dict) -> dict:
        """Execute tools with MCMC interception for cropping."""
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

                image_key_to_check = tool_args.get("image_url", state["original_figure_url"])

                if image_key_to_check in url_map:
                    image_url_to_use = url_map[image_key_to_check]
                else:
                    raise ValueError(
                        f"The image key '{image_key_to_check}' does not exist in the url_map. "
                        f"Available keys are: {list(url_map.keys())}"
                    )

                tool_args["image_url"] = image_url_to_use

                original_coords = None
                modified_coords = None
                if tool_name == "cropping":
                    original_coords = tool_args["coordinates"].copy()
                    validated_coords = validate_crop_coordinates(original_coords)
                    try:
                        original_image = await download_image_to_pil(state["original_figure_url"])
                        if image_url_to_use == state["original_figure_url"]:
                            image_to_crop = None
                        else:
                            image_to_crop = await download_image_to_pil(image_url_to_use)
                    except Exception:
                        logger.exception("mcmc_image_download_failed")
                        modified_coords = tool_args["coordinates"].copy()
                    else:
                        try:
                            bbox = await mcmc.get_action(
                                chat_model,
                                state["question"],
                                original_image,
                                image_to_crop,
                                validated_coords,
                                resolvability_sys_inst,
                            )
                        except Exception:
                            logger.exception("mcmc_policy_failed")
                            modified_coords = tool_args["coordinates"].copy()
                        else:
                            modified_coords = list(bbox)
                            tool_args["coordinates"] = modified_coords

                if tool_name in COORDINATE_INPUT_TOOLS:
                    tool_args = await _convert_to_absolute_coords(
                        state, tool_args, image_url_to_use, tool_name
                    )

                async with get_api_limiter():
                    observation = await asyncio.wait_for(
                        tool.ainvoke(tool_args), timeout=DEFAULT_TIMEOUT
                    )

                latency = time.perf_counter() - start_time

                if not observation.content or observation.content[0].type != "text":
                    raise Exception("Tool returned invalid or non-text content")

                tool_output_content = observation.content[0].text

                if isinstance(tool_output_content, dict):
                    data = tool_output_content
                elif isinstance(tool_output_content, str):
                    data = json.loads(tool_output_content)
                else:
                    raise TypeError(f"Unknown tool output type: {type(tool_output_content)}")

                if "detection" in tool_name and data.get("status") == "success":
                    data = await _convert_to_relative_coords(state, data, image_url_to_use)

                if tool_name == "cropping" and data.get("status") == "success":
                    if original_coords is not None and modified_coords is not None:
                        data["original_coordinates"] = original_coords
                        data["mcmc_selected_coordinates"] = modified_coords

                real_image_url = data.get("output_image")

                if real_image_url and data.get("status") == "success":
                    fake_url = generate_fake_url(state, tool_name)
                    state["url_map"][fake_url] = real_image_url

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

        state["messages"].extend(tool_msgs)
        state["tool_usage"] = tool_usage
        return state

    return mcmc_tool_executor


@node_registry.register("boed_crop_tool_executor")
def get_boed_crop_tool_executor(
    executable_tools: dict[str, StructuredTool],
    chat_model: BaseChatModel,
    resolvability_prompt: Path | str = PROMPT_DIR / "boed_resolvability.md",
    scaling_factors: list[float] = [1.5, 1, 0.8],  # noqa: B006
    n_samples: int = 5,
    **kwargs,
) -> Node:
    del kwargs

    # Get system prompts
    resolvability_sys_inst = Path(resolvability_prompt).read_text()

    # Setup BOED
    boed = BoedPolicy(scaling_factors, n_samples)

    async def boed_tool_executor(state: dict) -> dict:
        """Execute tools with BOED interception for cropping."""
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

                # 1. Find the image URL
                image_key_to_check = tool_args.get("image_url", state["original_figure_url"])

                if image_key_to_check in url_map:
                    image_url_to_use = url_map[image_key_to_check]
                else:
                    raise ValueError(
                        f"The image key '{image_key_to_check}' does not exist in the url_map. "
                        f"Available keys are: {list(url_map.keys())}"
                    )

                tool_args["image_url"] = image_url_to_use

                # 2. BOED INTERCEPTION for cropping BEFORE coordinate conversion
                original_coords = None
                modified_coords = None
                if tool_name == "cropping":
                    original_coords = tool_args["coordinates"].copy()
                    validated_coords = validate_crop_coordinates(original_coords)
                    try:
                        original_image = await download_image_to_pil(state["original_figure_url"])
                        if image_url_to_use == state["original_figure_url"]:
                            image_to_crop = None
                        else:
                            image_to_crop = await download_image_to_pil(image_url_to_use)
                    except Exception:
                        logger.exception("boed_image_download_failed")
                        modified_coords = tool_args["coordinates"].copy()
                    else:
                        bbox = await boed.get_action(
                            chat_model,
                            state["question"],
                            original_image,
                            image_to_crop,
                            validated_coords,
                            resolvability_sys_inst,
                        )
                        modified_coords = list(bbox)
                        tool_args["coordinates"] = modified_coords

                # 3. INBOUND coordinate conversion (Agent -> Tool)
                if tool_name in COORDINATE_INPUT_TOOLS:
                    tool_args = await _convert_to_absolute_coords(
                        state, tool_args, image_url_to_use, tool_name
                    )

                # 4. Run the tool
                async with get_api_limiter():
                    observation = await asyncio.wait_for(
                        tool.ainvoke(tool_args), timeout=DEFAULT_TIMEOUT
                    )

                latency = time.perf_counter() - start_time

                # 5. Parse the CallToolResult object
                if not observation.content or observation.content[0].type != "text":
                    raise Exception("Tool returned invalid or non-text content")

                tool_output_content = observation.content[0].text

                if isinstance(tool_output_content, dict):
                    data = tool_output_content
                elif isinstance(tool_output_content, str):
                    data = json.loads(tool_output_content)
                else:
                    raise TypeError(f"Unknown tool output type: {type(tool_output_content)}")

                if "detection" in tool_name and data.get("status") == "success":
                    data = await _convert_to_relative_coords(state, data, image_url_to_use)

                # 7. Inject BOED-modified coordinates for cropping tool
                if tool_name == "cropping" and data.get("status") == "success":
                    if original_coords is not None and modified_coords is not None:
                        data["original_coordinates"] = original_coords
                        data["boed_selected_coordinates"] = modified_coords

                real_image_url = data.get("output_image")

                if real_image_url and data.get("status") == "success":
                    fake_url = generate_fake_url(state, tool_name)
                    state["url_map"][fake_url] = real_image_url

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

                # Canonical log line
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

        # Update state
        state["messages"].extend(tool_msgs)
        state["tool_usage"] = tool_usage
        return state

    return boed_tool_executor


def validate_crop_coordinates(coords: list[int | float], min_size: int = 50) -> Bbox:
    # Round floats
    x1, y1, x2, y2 = map(round, coords)

    # Check positive area
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid coordinates (x2 <= x1 or y2 <= y1): {x1, y1, x2, y2}")

    # Check minimum size
    width = x2 - x1
    height = y2 - y1
    if width < min_size or height < min_size:
        raise ValueError(
            f"Invalid coordinates (width < {min_size} or height < {min_size}): {x1, y1, x2, y2}"
        )

    return (x1, y1, x2, y2)
