import cv2 as cv
import numpy as np
import os
import re


DONKEY_ROI = (0.118, 0.179, 0.079, 0.051)
CAR_ROI    = (0.753, 0.179, 0.079, 0.051)


def roi_to_pixels(frame, roi_rel):
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


def crop_roi(frame, roi):
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]


def extract_score_rois(frame):
    donkey_roi = roi_to_pixels(frame, DONKEY_ROI)
    car_roi = roi_to_pixels(frame, CAR_ROI)
    donkey_img = crop_roi(frame, donkey_roi)
    car_img = crop_roi(frame, car_roi)
    return donkey_img, car_img


def draw_score_rois(frame):
    donkey_x, donkey_y, donkey_w, donkey_h = roi_to_pixels(frame, DONKEY_ROI)
    car_x, car_y, car_w, car_h = roi_to_pixels(frame, CAR_ROI)

    cv.rectangle(
        frame,
        (donkey_x, donkey_y),
        (donkey_x + donkey_w, donkey_y + donkey_h),
        (255, 255, 0),
        2,
    )
    cv.putText(
        frame,
        "DONKEY ROI",
        (donkey_x, max(15, donkey_y - 5)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 0),
        1,
        cv.LINE_AA,
    )

    cv.rectangle(
        frame,
        (car_x, car_y),
        (car_x + car_w, car_y + car_h),
        (0, 255, 255),
        2,
    )
    cv.putText(
        frame,
        "CAR ROI",
        (car_x, max(15, car_y - 5)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv.LINE_AA,
    )


def preprocess_score_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Keep foreground (digits) white for contour extraction.
    if np.count_nonzero(binary) > (binary.size // 2):
        binary = cv.bitwise_not(binary)

    return binary


def crop_to_content(binary_img):
    ys, xs = np.where(binary_img > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return binary_img[y1:y2 + 1, x1:x2 + 1]


def load_score_templates(score_templates_dir: str) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}

    # Preferred format: digit_0.png ... digit_9.png
    for digit in range(10):
        path = os.path.join(score_templates_dir, f"digit_{digit}.png")
        image = cv.imread(path)
        if image is not None:
            processed = preprocess_score_image(image)
            cropped = crop_to_content(processed)
            if cropped is not None:
                templates[digit] = cropped

    if templates:
        return templates

    if not os.path.isdir(score_templates_dir):
        return templates

    # Fallback format: donkey_score_YYYYMMDD_HHMMSS_0000.png ... _0009.png
    for name in sorted(os.listdir(score_templates_dir)):
        match = re.match(r"donkey_score_\d{8}_\d{6}_(\d{4})\.png$", name)
        if not match:
            continue

        digit = int(match.group(1))
        if not (0 <= digit <= 9):
            continue

        path = os.path.join(score_templates_dir, name)
        image = cv.imread(path)
        if image is None:
            continue

        processed = preprocess_score_image(image)
        cropped = crop_to_content(processed)
        if cropped is not None:
            templates[digit] = cropped

    return templates


def split_score_digits(score_roi_img, max_digits: int = 3):
    binary = preprocess_score_image(score_roi_img)

    # Remove tiny specks before contour extraction.
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))

    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    roi_h, roi_w = binary.shape[:2]
    boxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        area = w * h

        if area < 6:
            continue
        if h < max(4, int(roi_h * 0.35)):
            continue
        if w > int(roi_w * 0.9) and h > int(roi_h * 0.9):
            # Skip large full-ROI blobs (background artifacts).
            continue

        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])
    if len(boxes) > max_digits:
        boxes = boxes[-max_digits:]

    digit_images = []
    for x, y, w, h in boxes:
        digit_bin = binary[y:y + h, x:x + w]
        digit_cropped = crop_to_content(digit_bin)
        if digit_cropped is not None:
            digit_images.append(digit_cropped)

    return digit_images

def predict_score_digit(digit_img, templates: dict[int, np.ndarray]):
    if not templates:
        return None, 0.0

    best_digit = None
    best_score = -1.0
    for digit, template_bin in templates.items():
        resized = cv.resize(digit_img, (template_bin.shape[1], template_bin.shape[0]), interpolation=cv.INTER_NEAREST)
        match = cv.matchTemplate(resized, template_bin, cv.TM_CCOEFF_NORMED)
        score = float(match[0, 0])
        if score > best_score:
            best_score = score
            best_digit = digit

    return best_digit, best_score


def predict_score_value(score_roi_img, templates: dict[int, np.ndarray]):
    digit_images = split_score_digits(score_roi_img, max_digits=3)
    if not digit_images:
        return None, 0.0

    digits = []
    scores = []
    for digit_img in digit_images:
        pred_digit, pred_score = predict_score_digit(digit_img, templates)
        if pred_digit is None:
            return None, 0.0
        digits.append(str(pred_digit))
        scores.append(pred_score)

    value = int("".join(digits))
    confidence = float(min(scores)) if scores else 0.0
    return value, confidence


def read_score_counters(frame, templates: dict[int, np.ndarray]):
    donkey_img, car_img = extract_score_rois(frame)
    donkey_value, donkey_score = predict_score_value(donkey_img, templates)
    car_value, car_score = predict_score_value(car_img, templates)
    return {
        "donkey": donkey_value,
        "driver": car_value,
        "donkey_conf": donkey_score,
        "driver_conf": car_score,
    }

def detect_one(
    frame,
    template_path: str,
    label: str,
    threshold: float = 0.8,
    color: tuple = (0, 255, 0),
) -> dict:
    template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Template not found: {template_path}")

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]

    print(f"width: {w}, height: {h}")
    res = cv.matchTemplate(frame_gray, template, cv.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv.minMaxLoc(res)

    found = max_val >= threshold
    result = {
        "label": label,
        "found": found,
        "score": float(max_val),
        "top_left": None,
        "center": None,
        "size": (w, h),
    }


    if found:
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        center = (top_left[0] + w // 2, top_left[1] + h // 2)

        cv.rectangle(frame, top_left, bottom_right, color, 2)
        cv.putText(
            frame,
            f"{label}: {max_val:.3f}",
            (top_left[0], max(20, top_left[1] - 10)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv.LINE_AA,
        )

        result["top_left"] = top_left
        result["center"] = center

    return result


def build_state(
    player_result: dict,
    donkey_result: dict,
    frame_shape: tuple,
    prev_state: np.ndarray | None = None,
) -> np.ndarray:
    height, width = frame_shape[:2]

    def _coords(result):
        if result["found"]:
            cx, cy = result["center"]
            return cx, cy, cx / width, cy / height
        return None, None, None, None

    px, py, px_n, py_n = _coords(player_result)
    dx, dy, dx_n, dy_n = _coords(donkey_result)
    rel_x = float((dx_n - px_n) if (px_n is not None and dx_n is not None) else 0.0)
    rel_y = float((dy_n - py_n) if (py_n is not None and dy_n is not None) else 0.0)

    prev_rel_x = 0.0
    prev_rel_y = 0.0
    if prev_state is not None and len(prev_state) >= 8:
        prev_rel_x = float(prev_state[6])
        prev_rel_y = float(prev_state[7])

    # Relative velocity between consecutive frames.
    rel_vx = rel_x - prev_rel_x
    rel_vy = rel_y - prev_rel_y

    state = np.array([
        float(player_result["found"]),
        float(donkey_result["found"]),
        float(px_n if px_n is not None else 0.0),
        float(py_n if py_n is not None else 0.0),
        float(dx_n if dx_n is not None else 0.0),
        float(dy_n if dy_n is not None else 0.0),
        rel_x,
        rel_y,
        rel_vx,
        rel_vy,
    ], dtype=np.float32)

    return state