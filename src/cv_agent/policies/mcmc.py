import random
import re

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import AnyMessage
from langfuse import observe
from PIL.Image import Image

from cv_agent.constants import Bbox
from cv_agent.models.openai import RateLimitedChatModel
from cv_agent.policies.events import (
    BATCH_REQUEST_COMPLETE,
    BATCH_REQUEST_ERROR,
    INVALID_CONFIDENCE_ASSESS,
    NODE_EVALUATED,
)
from cv_agent.policies.utils import ConfidenceEvalResult
from cv_agent.utils.qwen_vl import get_base64_url_for_qwen_vl, rel_coords_to_abs

logger = structlog.get_logger(__name__)


class McmcPolicy:
    def __init__(
        self,
        n_iterations: int = 6,
        n_samples: int = 3,
        eta: float = 0.15,
        acceptance_floor: float = 0.1,
        jitter_ratio: float = 0.05,
        min_size: int = 50,
    ) -> None:
        self.n_iterations = n_iterations
        self.n_samples = n_samples
        self.eta = eta
        self.acceptance_floor = acceptance_floor
        self.jitter_ratio = jitter_ratio
        self.min_size = min_size

    async def get_action(
        self,
        chat_model: BaseChatModel | RateLimitedChatModel,
        question: str,
        original_image: Image,
        image_to_crop: Image | None,
        proposed_bbox: Bbox,
        resolvability_sys_inst: str,
    ) -> Bbox:
        current_bbox = proposed_bbox
        current_eval = await evaluate_candidate_confidence(
            chat_model,
            question,
            original_image,
            image_to_crop,
            current_bbox,
            resolvability_sys_inst,
            self.n_samples,
        )
        current_score = current_eval["score"]
        logger.info(NODE_EVALUATED, bbox=f"{current_bbox}", **current_eval)

        best_bbox = current_bbox
        best_score = current_score

        if best_score >= 1.0:
            logger.info("mcmc_complete", best_bbox=f"{best_bbox}", best_score=best_score)
            return best_bbox

        for _ in range(self.n_iterations):
            proposal_bbox = propose_bbox(
                current_bbox,
                eta=self.eta,
                jitter_ratio=self.jitter_ratio,
                min_size=self.min_size,
            )
            proposal_eval = await evaluate_candidate_confidence(
                chat_model,
                question,
                original_image,
                image_to_crop,
                proposal_bbox,
                resolvability_sys_inst,
                self.n_samples,
            )
            proposal_score = proposal_eval["score"]
            logger.info(NODE_EVALUATED, bbox=f"{proposal_bbox}", **proposal_eval)

            if should_accept_proposal(
                proposal_score,
                current_score,
                acceptance_floor=self.acceptance_floor,
            ):
                current_bbox = proposal_bbox
                current_score = proposal_score
                logger.info("mcmc_move_accepted", bbox=f"{current_bbox}", score=current_score)

            if current_score > best_score:
                best_bbox = current_bbox
                best_score = current_score

            if best_score >= 1.0:
                break

        logger.info("mcmc_complete", best_bbox=f"{best_bbox}", best_score=best_score)
        return best_bbox


def propose_bbox(
    bbox: Bbox,
    eta: float,
    jitter_ratio: float,
    min_size: int = 50,
    bounds: int = 1000,
) -> Bbox:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    center_x = (x1 + x2) / 2 + random.gauss(0, width * eta)
    center_y = (y1 + y2) / 2 + random.gauss(0, height * eta)
    next_width = width + random.gauss(0, width * jitter_ratio)
    next_height = height + random.gauss(0, height * jitter_ratio)

    return clamp_bbox(
        center_x,
        center_y,
        next_width,
        next_height,
        min_size=min_size,
        bounds=bounds,
    )


def clamp_bbox(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    min_size: int = 50,
    bounds: int = 1000,
) -> Bbox:
    bounded_width = int(round(max(min_size, min(bounds, width))))
    bounded_height = int(round(max(min_size, min(bounds, height))))

    x1 = int(round(center_x - bounded_width / 2))
    y1 = int(round(center_y - bounded_height / 2))
    x2 = x1 + bounded_width
    y2 = y1 + bounded_height

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > bounds:
        x1 -= x2 - bounds
        x2 = bounds
    if y2 > bounds:
        y1 -= y2 - bounds
        y2 = bounds

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(bounds, x2)
    y2 = min(bounds, y2)

    if x2 - x1 < min_size:
        x2 = min(bounds, x1 + min_size)
        x1 = max(0, x2 - min_size)
    if y2 - y1 < min_size:
        y2 = min(bounds, y1 + min_size)
        y1 = max(0, y2 - min_size)

    return (x1, y1, x2, y2)


def should_accept_proposal(
    proposal_score: float,
    current_score: float,
    acceptance_floor: float,
) -> bool:
    return proposal_score >= current_score or random.random() < acceptance_floor


@observe(name="mcmc-confidence", capture_input=False)
async def evaluate_candidate_confidence(
    chat_model: BaseChatModel | RateLimitedChatModel,
    question: str,
    original_image: Image,
    image_to_crop: Image | None,
    bbox: Bbox,
    sys_inst: str,
    n_samples: int,
) -> ConfidenceEvalResult:
    prompt = construct_input_for_confidence(
        question,
        original_image,
        image_to_crop,
        bbox,
        sys_inst,
    )
    batch_inputs = [prompt] * n_samples
    try:
        responses = await chat_model.abatch(batch_inputs)
    except Exception:
        logger.exception(BATCH_REQUEST_ERROR)
        return {"score": 0.0, "votes_yes": 0, "total_samples": 0}

    logger.info(BATCH_REQUEST_COMPLETE)

    votes_yes = 0
    total = 0
    for response in responses:
        if not isinstance(response.content, str):
            logger.warning(INVALID_CONFIDENCE_ASSESS, fallback="no")
            continue

        total += 1
        answer = extract_binary_answer(response.content)
        if answer == "yes":
            votes_yes += 1

    score = votes_yes / total if total > 0 else 0.0
    return {"score": score, "votes_yes": votes_yes, "total_samples": total}


def construct_input_for_confidence(
    question: str,
    original_image: Image,
    image_to_crop: Image | None,
    bbox: Bbox,
    sys_inst: str,
) -> list[AnyMessage]:
    labels, urls = [], []
    labels.append("original image (provided with the question)")
    urls.append(get_base64_url_for_qwen_vl(original_image))

    if image_to_crop is not None:
        labels.append("image in tool args (agent wants to crop this)")
        crop_source = image_to_crop
        urls.append(get_base64_url_for_qwen_vl(image_to_crop))
    else:
        crop_source = original_image

    labels.append("cropped candidate region")
    urls.append(
        get_base64_url_for_qwen_vl(
            crop_source.crop(rel_coords_to_abs(bbox, crop_source.size)),  # type: ignore[arg-type]
        )
    )

    content = [{"type": "text", "text": f"**Original Question:**\n{question}\n\n"}]
    for label, url in zip(labels, urls, strict=True):
        content.append({"type": "text", "text": f"Here is the {label}:"})
        content.append({"type": "image_url", "image_url": {"url": url}})  # type: ignore[arg-type]

    return [SystemMessage(content=sys_inst), HumanMessage(content=content)]  # type: ignore[arg-type]


def extract_binary_answer(content: str) -> str:
    normalized = content.strip()
    match = re.search(r"<answer>(.*?)</answer>", normalized, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip().lower()
        if "yes" in answer:
            return "yes"
        if "no" in answer:
            return "no"

    if "yes" in normalized.lower():
        return "yes"
    return "no"
