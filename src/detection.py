import cv2 as cv
import numpy as np


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


def build_state(player_result: dict, donkey_result: dict, frame_shape: tuple) -> dict:
    height, width = frame_shape[:2]

    def _coords(result):
        if result["found"]:
            cx, cy = result["center"]
            return cx, cy, cx / width, cy / height
        return None, None, None, None

    px, py, px_n, py_n = _coords(player_result)
    dx, dy, dx_n, dy_n = _coords(donkey_result)

    state = {
        "player_found": int(player_result["found"]),
        "donkey_found": int(donkey_result["found"]),
        "player_x": px,
        "player_y": py,
        "player_x_norm": px_n,
        "player_y_norm": py_n,
        "donkey_x": dx,
        "donkey_y": dy,
        "donkey_x_norm": dx_n,
        "donkey_y_norm": dy_n,
        "dx": (dx - px) if (px is not None and dx is not None) else None,
        "dy": (dy - py) if (py is not None and dy is not None) else None,
    }

    return state