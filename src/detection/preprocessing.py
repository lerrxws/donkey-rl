import cv2 as cv
import numpy as np
from src.config import (
    DIGIT_NORMALIZED_SIZE,
    DIGIT_PAD,
    SATURATION_THRESHOLD,
    MIN_BRIGHTNESS_OFFSET,
    MIN_BRIGHTNESS_FLOOR,
)


def preprocess_score_image(image: np.ndarray) -> np.ndarray | None:
    if image is None or image.size == 0:
        return None

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1]
    grayish_mask = saturation < SATURATION_THRESHOLD

    vals = gray[grayish_mask]
    if vals.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)

    bg_level = float(np.median(vals))
    p75 = float(np.percentile(vals, 75))
    threshold = max(bg_level + MIN_BRIGHTNESS_OFFSET, p75, MIN_BRIGHTNESS_FLOOR)

    mask = (grayish_mask & (gray >= threshold)).astype(np.uint8) * 255

    kernel = np.ones((2, 2), dtype=np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    return mask


def crop_to_content(binary_img: np.ndarray) -> np.ndarray | None:
    if binary_img is None or binary_img.size == 0:
        return None

    ys, xs = np.where(binary_img > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    return binary_img[y1:y2 + 1, x1:x2 + 1]


def normalize_digit(
    binary_img: np.ndarray,
    size: tuple[int, int] = DIGIT_NORMALIZED_SIZE,
    pad: int = DIGIT_PAD,
) -> np.ndarray | None:
    cropped = crop_to_content(binary_img)

    if cropped is None:
        return None

    target_h, target_w = size
    h, w = cropped.shape[:2]

    if h <= 0 or w <= 0:
        return None

    max_w = target_w - 2 * pad
    max_h = target_h - 2 * pad
    scale = min(max_w / w, max_h / h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv.resize(cropped, (new_w, new_h), interpolation=cv.INTER_NEAREST)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas
