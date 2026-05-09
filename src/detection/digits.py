import cv2 as cv
import numpy as np
from config import (
    SEGMENT_MIN_WIDTH,
    SEGMENT_MERGE_GAP,
)
from src.detection.preprocessing import preprocess_score_image, crop_to_content, normalize_digit


def predict_score_value(
    score_roi_img: np.ndarray,
    templates: dict[int, np.ndarray],
    max_digits: int = 4,
    debug_prefix: str | None = None,
) -> tuple[int | None, float, int]:
    digit_images = _split_score_digits(
        score_roi_img,
        max_digits=max_digits,
        debug_prefix=debug_prefix,
    )

    if not digit_images:
        return None, 0.0, 0

    digits = []
    scores = []

    for digit_img in digit_images:
        pred_digit, pred_score = _predict_score_digit(digit_img, templates)

        if pred_digit is None:
            return None, 0.0, 0

        digits.append(str(pred_digit))
        scores.append(float(pred_score))

    confidence = min(scores) if scores else 0.0
    value = int("".join(digits))

    return value, confidence, len(digits)

def _dice_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)

    a_count = int(np.count_nonzero(a_bin))
    b_count = int(np.count_nonzero(b_bin))

    if a_count + b_count == 0:
        return 0.0

    intersection = int(np.count_nonzero((a_bin == 1) & (b_bin == 1)))

    return (2.0 * intersection) / (a_count + b_count)


def _split_score_digits(
    score_roi_img: np.ndarray,
    max_digits: int = 4,
    debug_prefix: str | None = None,
) -> list[np.ndarray]:
    binary = preprocess_score_image(score_roi_img)

    if binary is None:
        return []

    if debug_prefix is not None:
        cv.imwrite(f"{debug_prefix}_01_binary_raw.png", binary)

    content = crop_to_content(binary)

    if content is None:
        return []

    binary = content

    if debug_prefix is not None:
        cv.imwrite(f"{debug_prefix}_02_content.png", binary)

    h, w = binary.shape[:2]

    if h <= 0 or w <= 0:
        return []

    segments = _find_column_segments(binary)
    segments = _merge_segments(segments)

    if not segments:
        return []

    if len(segments) > max_digits:
        segments = segments[-max_digits:]

    digit_images = []
    debug_view = cv.cvtColor(binary, cv.COLOR_GRAY2BGR) if debug_prefix else None

    for i, (x1, x2) in enumerate(segments):
        digit_bin = binary[:, x1:x2]

        digit_content = crop_to_content(digit_bin)
        if digit_content is None:
            continue

        digit_norm = normalize_digit(digit_content)
        if digit_norm is None:
            continue

        digit_images.append(digit_norm)

        if debug_prefix is not None:
            cv.rectangle(debug_view, (x1, 0), (x2, h - 1), (0, 255, 0), 1)
            cv.imwrite(f"{debug_prefix}_digit_{i}.png", digit_norm)

    if debug_prefix is not None:
        cv.imwrite(f"{debug_prefix}_03_segments.png", debug_view)

    return digit_images


def _predict_score_digit(
    digit_img: np.ndarray,
    templates: dict[int, np.ndarray],
) -> tuple[int | None, float]:
    if not templates:
        return None, 0.0

    if digit_img is None or np.count_nonzero(digit_img) == 0:
        return None, 0.0

    best_digit = None
    best_score = -1.0

    for digit, template_bin in templates.items():
        if digit_img.shape != template_bin.shape:
            template_resized = cv.resize(
                template_bin,
                (digit_img.shape[1], digit_img.shape[0]),
                interpolation=cv.INTER_NEAREST,
            )
        else:
            template_resized = template_bin

        score = _dice_similarity(digit_img, template_resized)

        if score > best_score:
            best_score = float(score)
            best_digit = int(digit)

    return best_digit, best_score


def _find_column_segments(binary: np.ndarray) -> list[tuple[int, int]]:
    col_has_pixels = np.any(binary > 0, axis=0)
    w = binary.shape[1]

    segments = []
    in_segment = False
    start = 0

    for x, has_pixels in enumerate(col_has_pixels):
        if has_pixels and not in_segment:
            start = x
            in_segment = True
        elif not has_pixels and in_segment:
            segments.append((start, x))
            in_segment = False

    if in_segment:
        segments.append((start, w))

    return [(x1, x2) for x1, x2 in segments if (x2 - x1) >= SEGMENT_MIN_WIDTH]


def _merge_segments(segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not segments:
        return []

    merged = []
    cur_start, cur_end = segments[0]

    for x1, x2 in segments[1:]:
        gap = x1 - cur_end

        if gap <= SEGMENT_MERGE_GAP:
            cur_end = x2
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = x1, x2

    merged.append((cur_start, cur_end))

    return merged