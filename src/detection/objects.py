import cv2 as cv
import numpy as np
from src.config import TEMPLATE_MATCH_THRESHOLD


def detect_object(
    frame: np.ndarray,
    template_path: str,
    label: str,
    threshold: float = TEMPLATE_MATCH_THRESHOLD,
) -> dict:
    template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)

    if template is None:
        raise FileNotFoundError(f"Template not found: {template_path}")

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]

    res = cv.matchTemplate(frame_gray, template, cv.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv.minMaxLoc(res)

    found = max_val >= threshold

    if not found:
        return {
            "label":    label,
            "found":    False,
            "score":    float(max_val),
            "top_left": None,
            "center":   None,
            "size":     (w, h),
        }

    top_left = max_loc
    center = (top_left[0] + w // 2, top_left[1] + h // 2)

    return {
        "label":    label,
        "found":    True,
        "score":    float(max_val),
        "top_left": top_left,
        "center":   center,
        "size":     (w, h),
    }