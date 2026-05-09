import numpy as np
from src.config import (
    DONKEY_ROI_WIDE,
    CAR_ROI_WIDE,
)


def extract_score_rois(frame: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "donkey": _crop_roi(frame, _roi_to_pixels(frame, DONKEY_ROI_WIDE)),
        "car":    _crop_roi(frame, _roi_to_pixels(frame, CAR_ROI_WIDE)),
    }


def get_roi_debug_info(frame: np.ndarray) -> dict[str, dict]:
    return {
        "donkey": dict(zip(("x", "y", "w", "h"), _roi_to_pixels(frame, DONKEY_ROI_WIDE))),
        "car":    dict(zip(("x", "y", "w", "h"), _roi_to_pixels(frame, CAR_ROI_WIDE))),
    }

def _roi_to_pixels(frame: np.ndarray, roi_rel: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    rx, ry, rw, rh = roi_rel

    x = int(round(rx * width))
    y = int(round(ry * height))
    w = max(1, int(round(rw * width)))
    h = max(1, int(round(rh * height)))

    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = min(w, width - x)
    h = min(h, height - y)

    return x, y, w, h


def _crop_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]