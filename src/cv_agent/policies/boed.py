import re

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langfuse import observe
from PIL.Image import Image

from cv_agent.constants import Bbox
from cv_agent.policies.events import (
    BATCH_REQUEST_COMPLETE,
    BATCH_REQUEST_ERROR,
    BOED_COMPLETE,
    INVALID_CONFIDENCE_ASSESS,
    NODE_EVALUATED,
)
from cv_agent.policies.utils import ConfidenceEvalResult, generate_candidates
from cv_agent.utils.qwen_vl import get_base64_url_for_qwen_vl, rel_coords_to_abs

logger = structlog.get_logger(__name__)


class BoedPolicy:
    def __init__(self, scaling_factors: list[float], n_samples: int) -> None:
        self.scaling_factors = scaling_factors
        self.n_samples = n_samples

    async def get_action(
        self,
        chat_model: BaseChatModel,
        question: str,
        original_image: Image,
        image_to_crop: Image | None,
        proposed_bbox: Bbox,
        resolvability_sys_inst: str,
    ) -> Bbox:
        if image_to_crop is None:
            i2c = original_image
        else:
            i2c = image_to_crop

        coords_candidates = generate_candidates(proposed_bbox, self.scaling_factors)
        values = []
        for coords_candidate in coords_candidates:
            image_cropped = i2c.crop(rel_coords_to_abs(coords_candidate, i2c.size))  # type: ignore[arg-type]
            evaluation = await evaluate_candidate_confidence(
                chat_model,
                question,
                original_image,
                image_to_crop,
                image_cropped,
                resolvability_sys_inst,
                self.n_samples,
            )
            values.append(evaluation["score"])
            logger.info(NODE_EVALUATED, bbox=f"{coords_candidate}", **evaluation)

        _, best_bbox = max(zip(values, coords_candidates, strict=True), key=lambda item: item[0])
        logger.info(BOED_COMPLETE, best_bbox=f"{best_bbox}", values=f"{values}")
        return best_bbox


@observe(name="boed-confidence", capture_input=False)
async def evaluate_candidate_confidence(
    chat_model: BaseChatModel,
    question: str,
    original_image: Image,
    image_to_crop: Image | None,
    image_cropped: Image,
    sys_inst: str,
    n_samples: int,
) -> ConfidenceEvalResult:
    prompt = construct_input_for_confidence(
        question, original_image, image_to_crop, image_cropped, sys_inst
    )
    batch_inputs = [prompt] * n_samples
    try:
        # NOTE: this bypasses the API rate limiter
        responses = await chat_model.abatch(batch_inputs)
    except Exception:
        logger.exception(BATCH_REQUEST_ERROR)
        return {"score": 0.0, "votes_yes": 0, "total_samples": 0}

    logger.info(BATCH_REQUEST_COMPLETE)

    votes_yes, total = 0, 0
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

    score = votes_yes / total if total > 0 else 0.0
    return {"score": score, "votes_yes": votes_yes, "total_samples": total}


def construct_input_for_confidence(
    question: str,
    original_image: Image,
    image_to_crop: Image | None,
    image_cropped: Image,
    sys_inst,
) -> list[AnyMessage]:
    labels, urls = [], []
    # Original image (always occurs in prompt)
    labels.append("original image (provided with the question)")
    urls.append(get_base64_url_for_qwen_vl(original_image))
    # Image to crop
    if image_to_crop is not None:
        labels.append("image in tool args (agent wants to crop this)")
        urls.append(get_base64_url_for_qwen_vl(image_to_crop))
    else:
        image_to_crop = original_image
    # Crop candidate
    labels.append("cropped candidate region")
    urls.append(get_base64_url_for_qwen_vl(image_cropped))

    content = [{"type": "text", "text": f"**Original Question:**\n{question}\n\n"}]
    for label, url in zip(labels, urls, strict=True):
        content.append({"type": "text", "text": f"Here is the {label}:"})
        content.append({"type": "image_url", "image_url": {"url": url}})  # type: ignore[arg-type]

    messages = [SystemMessage(content=sys_inst), HumanMessage(content=content)]  # type: ignore[arg-type]
    return messages
