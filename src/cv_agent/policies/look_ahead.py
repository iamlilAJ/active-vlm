import re
from typing import TypedDict

import structlog
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langfuse import observe
from langgraph.graph.message import AnyMessage
from PIL.Image import Image

from cv_agent.constants import Bbox
from cv_agent.models.openai import RateLimitedChatModel
from cv_agent.policies.events import (
    BATCH_REQUEST_COMPLETE,
    BATCH_REQUEST_ERROR,
    BINDED_LLM_CALL_FAILED,
    BINDED_LLM_NO_TOOL_CALL,
    INVALID_CONFIDENCE_ASSESS,
    INVALID_EXPANSION,
    INVALID_IMAGE_REF,
    LOOK_AHEAD_COMPLETE,
    NODE_EVALUATED,
)
from cv_agent.policies.utils import ConfidenceEvalResult, generate_candidates
from cv_agent.utils.qwen_vl import get_base64_url_for_qwen_vl, rel_coords_to_abs

logger = structlog.get_logger(__name__)


class LookAheadPolicy:
    def __init__(self, scaling_factors: list[float], n_samples: int) -> None:
        self.scaling_factors = scaling_factors
        self.n_samples = n_samples

    async def get_action(
        self,
        chat_model: BaseChatModel | RateLimitedChatModel,
        question: str,
        original_image: Image,
        image_to_crop: Image | None,
        proposed_bbox: Bbox,
        inner_crop_sys_inst: str,
        resolvability_sys_inst: str,
    ) -> Bbox:
        if image_to_crop is None:
            i2c = original_image
        else:
            i2c = image_to_crop

        model_binded = chat_model.bind_tools([crop])
        coords_candidates = generate_candidates(proposed_bbox, self.scaling_factors)
        values = []
        for coords_candidate in coords_candidates:
            image_cropped = i2c.crop(rel_coords_to_abs(coords_candidate, i2c.size))  # type: ignore[arg-type]

            expansion = await look_ahead_expansion(
                model_binded,
                question,
                original_image,
                image_to_crop,
                image_cropped,
                inner_crop_sys_inst,
            )
            if expansion is None:
                logger.warning(INVALID_EXPANSION, fallback=0)
                values.append(0.0)
            else:
                image_ref = expansion["image"].lower()
                if "original" in image_ref:
                    sub_i2c = original_image
                elif "image_1" in image_ref:
                    sub_i2c = image_to_crop or original_image
                elif "image_2" in image_ref:
                    sub_i2c = image_cropped
                else:
                    logger.warning(INVALID_IMAGE_REF, image=image_ref, fallback="original")
                    sub_i2c = original_image

                sub_candidates = generate_candidates(expansion["coordinates"], self.scaling_factors)
                confidences = []
                for sub_coords in sub_candidates:
                    sub_image = sub_i2c.crop(rel_coords_to_abs(sub_coords, sub_i2c.size))  # type: ignore[arg-type]
                    evaluation = await evaluate_candidate_confidence(
                        chat_model,
                        question,
                        original_image,
                        image_cropped,
                        sub_image,
                        resolvability_sys_inst,
                        self.n_samples,
                    )
                    confidences.append(evaluation["score"])
                    logger.info(NODE_EVALUATED, bbox=f"{sub_coords}", **evaluation)

                values.append(sum(confidences) / len(confidences))

        _, best_bbox = max(zip(values, coords_candidates, strict=True), key=lambda item: item[0])
        logger.info(LOOK_AHEAD_COMPLETE, best_bbox=f"{best_bbox}", values=f"{values}")
        return best_bbox


class ExpansionResult(TypedDict):
    image: str
    coordinates: Bbox


@observe(name="look-ahead-expansion", capture_input=False)
async def look_ahead_expansion(
    binded_model, question, original_image, image_to_crop, image_cropped, sys_inst
) -> ExpansionResult | None:
    prompt = construct_input_for_binded_model(
        question, original_image, image_to_crop, image_cropped, sys_inst
    )
    try:
        response = await binded_model.ainvoke(prompt)
    except Exception:
        logger.exception(BINDED_LLM_CALL_FAILED)
        return None

    assert isinstance(response, AIMessage)
    if not response.tool_calls:
        logger.warning(BINDED_LLM_NO_TOOL_CALL)
        return None

    tool_args = response.tool_calls[0]["args"]

    return {"image": tool_args["image"], "coordinates": tuple(tool_args["coordinates"])}


def construct_input_for_confidence(
    question: str, original_image: Image, image_to_crop: Image, image_cropped: Image, sys_inst: str
) -> list[AnyMessage]:
    labels, urls = [], []
    # Original image (always occurs in prompt)
    labels.append("original image (provided with the question)")
    urls.append(get_base64_url_for_qwen_vl(original_image))
    # Image to crop
    labels.append("image in tool args (agent wants to crop this)")
    urls.append(get_base64_url_for_qwen_vl(image_to_crop))
    # Crop candidate
    labels.append("cropped candidate region")
    urls.append(get_base64_url_for_qwen_vl(image_cropped))

    content = [{"type": "text", "text": f"**Original Question:**\n{question}\n\n"}]
    for label, url in zip(labels, urls, strict=True):
        content.append({"type": "text", "text": f"Here is the {label}:"})
        content.append({"type": "image_url", "image_url": {"url": url}})  # type: ignore[arg-type]

    messages = [SystemMessage(content=sys_inst), HumanMessage(content=content)]  # type: ignore[arg-type]
    return messages


@observe(name="look-ahead-confidence", capture_input=False)
async def evaluate_candidate_confidence(
    chat_model: BaseChatModel | RateLimitedChatModel,
    question: str,
    original_image: Image,
    image_cropped: Image,
    sub_image: Image,
    sys_inst: str,
    n_samples: int,
) -> ConfidenceEvalResult:
    prompt = construct_input_for_confidence(
        question,
        original_image,
        image_cropped,
        sub_image,
        sys_inst,
    )
    batch_inputs = [prompt] * n_samples
    try:
        # NOTE: this bypasses the API rate limiter
        responses = await chat_model.abatch(batch_inputs)
    except Exception:
        logger.exception(BATCH_REQUEST_ERROR)
        return {"score": 0.0, "votes_yes": 0, "total_samples": 0}

    logger.info(BATCH_REQUEST_COMPLETE)

    # NOTE: we won't deal with cases where len(responses) != n_samples
    votes_yes = 0
    total = 0
    for response in responses:
        if not isinstance(response.content, str):
            logger.warning(INVALID_CONFIDENCE_ASSESS, fallback="no")
        else:
            total += 1
            content_str = response.content.strip()
            match = re.search(r"<answer>(.*?)</answer>", content_str, re.DOTALL | re.IGNORECASE)
            if match:
                answer = match.group(1).strip().lower()
                if "yes" in answer:
                    votes_yes += 1
            else:
                # Fallback: check if "yes" appears in content
                if "yes" in content_str.lower():
                    votes_yes += 1

    score = votes_yes / total
    return {"score": score, "votes_yes": votes_yes, "total_samples": total}


def construct_input_for_binded_model(
    question: str,
    original_image: Image,
    image_to_crop: Image | None,
    candidate_image: Image,
    sys_inst: str,
) -> list[AnyMessage]:
    labels, urls = [], []
    # Original image (always occurs in prompt)
    labels.append("original image (provided with the question, you can refer to it as `original`)")
    urls.append(get_base64_url_for_qwen_vl(original_image))
    # Context images
    cnt = 1
    if image_to_crop is not None:
        labels.append(f"history image {cnt} (you can refer to it as `image_{cnt}`)")
        urls.append(get_base64_url_for_qwen_vl(image_to_crop))
        cnt += 1
    else:
        image_to_crop = original_image
    # Crop candidate
    labels.append(f"history image {cnt} (you can refer to it as `image_{cnt}`)")
    urls.append(get_base64_url_for_qwen_vl(candidate_image))

    content = [{"type": "text", "text": f"**Original Question:**\n{question}\n\n"}]
    for label, url in zip(labels, urls, strict=True):
        content.append({"type": "text", "text": f"Here is the {label}:"})
        content.append({"type": "image_url", "image_url": {"url": url}})  # type: ignore[arg-type]

    messages = [SystemMessage(content=sys_inst), HumanMessage(content=content)]  # type: ignore[arg-type]
    return messages


@tool
def crop(image: str, coordinates: list[int]) -> dict:
    """Crop the given image based on the provided coordinates (x1, y1, x2, y2)."""
    # a placeholder
    del image, coordinates
    return {"result": "A placeholder"}
