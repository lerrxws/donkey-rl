import cv2 as cv
import numpy as np


DONKEY_ROI = (0.116, 0.179, 0.079, 0.051)
CAR_ROI    = (0.751, 0.179, 0.079, 0.051)


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