import os
import cv2 as cv
import numpy as np
from src.detection.roi import extract_score_rois
from src.detection.preprocessing import preprocess_score_image, normalize_digit
from src.detection.digits import predict_score_value


def load_score_templates(score_templates_dir: str) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}

    if not os.path.isdir(score_templates_dir):
        print(f"[WARN] Score templates dir not found: {score_templates_dir}")
        return templates

    for digit in range(10):
        path = os.path.join(score_templates_dir, f"digit_{digit}.png")
        image = cv.imread(path)

        if image is None:
            print(f"[WARN] Missing score template: {path}")
            continue

        processed = preprocess_score_image(image)
        normalized = normalize_digit(processed)

        if normalized is None:
            print(f"[WARN] Failed to normalize score template: {path}")
            continue

        templates[digit] = normalized

    print("Loaded score templates:", sorted(templates.keys()))

    if len(templates) < 10:
        print(f"[WARN] Loaded only {len(templates)}/10 score templates")

    return templates


def read_score_counters(
    frame: np.ndarray,
    templates: dict[int, np.ndarray],
    debug: bool = False,
) -> dict[str, int | float | None]:
    rois = extract_score_rois(frame)

    donkey_value, donkey_conf, _ = predict_score_value(
        rois["donkey"],
        templates,
        max_digits=4,
        debug_prefix="debug_donkey" if debug else None,
    )

    car_value, car_conf, _ = predict_score_value(
        rois["car"],
        templates,
        max_digits=4,
        debug_prefix="debug_car" if debug else None,
    )

    if debug:
        cv.imwrite("debug_donkey_raw.png", rois["donkey"])
        cv.imwrite("debug_car_raw.png", rois["car"])
        print(f"[SCORE DEBUG] donkey: {donkey_value} conf={donkey_conf:.3f}")
        print(f"[SCORE DEBUG] car:    {car_value} conf={car_conf:.3f}")

    return {
        "donkey":      donkey_value,
        "driver":      car_value,
        "donkey_conf": donkey_conf,
        "driver_conf": car_conf,
    }