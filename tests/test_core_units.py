from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

import cv_agent.policies.mcmc as mcmc
from cv_agent.core.registries import prompt_generator_registry, tool_registry
from cv_agent.core.state import create_agent_state
from cv_agent.utils.grader import Grade, grade_answer


class AgentState(TypedDict):
    task: str
    messages: Annotated[list[AnyMessage], add_messages]
    current_plan: list[dict[str, Any]]
    file_urls: dict[str, str]


def test_create_agent_state() -> None:
    state_definition = {
        "task": "str",
        "messages": "Annotated[list[AnyMessage], add_messages]",
        "current_plan": "list[dict[str, Any]]",
        "file_urls": "dict[str, str]",
    }

    agent_state = create_agent_state(state_definition)

    assert agent_state.__annotations__ == AgentState.__annotations__


@pytest.mark.parametrize(
    ("registry_name", "tool_name"),
    [
        ("ocr", "ocr"),
        ("detection", "detection"),
        ("detection_small_object", "detection_small_object"),
        ("segmentation", "segmentation"),
        ("cropping", "cropping"),
        ("depth_estimation", "depth_estimation"),
    ],
)
def test_vision_tools_registered_without_private_url(registry_name: str, tool_name: str) -> None:
    tool = tool_registry.get(registry_name, server_url="http://example.test/mcp")

    assert tool.name == tool_name


def test_planner_prompt_generator_loads_template() -> None:
    generator = prompt_generator_registry.get("planner")

    assert "Computer Vision" in generator({})


def test_grade_answer_for_mcq() -> None:
    assert grade_answer("<answer>A</answer>", "A") == Grade.CORRECT
    assert grade_answer("B", "A") == Grade.WRONG
    assert grade_answer("not sure", "A") == Grade.NO_ANSWER


def test_mcmc_bbox_proposal_stays_in_bounds() -> None:
    bbox = mcmc.propose_bbox((20, 20, 90, 90), eta=0.5, jitter_ratio=0.5, min_size=50)
    x1, y1, x2, y2 = bbox

    assert 0 <= x1 < x2 <= 1000
    assert 0 <= y1 < y2 <= 1000
    assert x2 - x1 >= 50
    assert y2 - y1 >= 50


def test_mcmc_acceptance_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    assert mcmc.should_accept_proposal(0.8, 0.3, acceptance_floor=0.0)

    monkeypatch.setattr(mcmc.random, "random", lambda: 0.05)
    assert mcmc.should_accept_proposal(0.2, 0.9, acceptance_floor=0.1)


def test_extract_binary_answer_defaults_to_no() -> None:
    assert mcmc.extract_binary_answer("<answer>yes</answer>") == "yes"
    assert mcmc.extract_binary_answer("unclear response") == "no"
