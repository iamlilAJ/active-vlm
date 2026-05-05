from typing import TypedDict

from cv_agent.constants import Bbox


class ConfidenceEvalResult(TypedDict):
    score: float
    votes_yes: int
    total_samples: int


def generate_candidates(bbox: Bbox, scaling_factors: list[float]) -> list[Bbox]:
    return [scale_bbox_uniformly(bbox, factor) for factor in scaling_factors]


def scale_bbox_uniformly(bbox: Bbox, scale_factor: float) -> Bbox:
    def avg(*args: int | float) -> float:
        return sum(args) / len(args)

    def half(n: int | float) -> float:
        return n / 2.0

    x1, y1, x2, y2 = bbox

    center_x = avg(x1, x2)
    center_y = avg(y1, y2)

    width = x2 - x1
    height = y2 - y1

    new_w = width * scale_factor
    new_h = height * scale_factor

    new_x1 = center_x - half(new_w)
    new_y1 = center_y - half(new_h)
    new_x2 = center_x + half(new_w)
    new_y2 = center_y + half(new_h)

    # Clamp to [0, 1000] and round
    res = tuple(map(round, (max(0, new_x1), max(0, new_y1), min(1000, new_x2), min(1000, new_y2))))

    return res  # type: ignore[arg-type]
