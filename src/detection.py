import cv2 as cv
import numpy as np
import os


# ============================================================
# SCORE ROI CONFIG
# ============================================================

# SMALL — старый ROI, хорошо подходит для 1–2 цифр.
# WIDE — расширенный вправо ROI, нужен для 3+ цифр.
#
# Если 100/999 всё ещё обрезается — увеличивай width у *_WIDE.
# Если попадает мусор — уменьши width или сдвинь x.
DONKEY_ROI_SMALL = (0.118, 0.179, 0.079, 0.051)
DONKEY_ROI_WIDE  = (0.118, 0.179, 0.130, 0.051)

CAR_ROI_SMALL = (0.753, 0.179, 0.079, 0.051)
CAR_ROI_WIDE  = (0.753, 0.179, 0.130, 0.051)

# Для обратной совместимости со старым кодом.
DONKEY_ROI = DONKEY_ROI_SMALL
CAR_ROI = CAR_ROI_SMALL

DIGIT_NORMALIZED_SIZE = (24, 16)  # (height, width)

# Внутренний порог. Не путать с MIN_CONF в game.py.
# Здесь лучше не делать слишком строго.
SCORE_INTERNAL_MIN_CONF = 0.15


# ============================================================
# ROI UTILS
# ============================================================

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
    return frame[y:y + h, x:x + w]


def extract_score_rois(frame):
    """
    Старый API: возвращает small ROI.
    """
    donkey_roi = roi_to_pixels(frame, DONKEY_ROI_SMALL)
    car_roi = roi_to_pixels(frame, CAR_ROI_SMALL)

    donkey_img = crop_roi(frame, donkey_roi)
    car_img = crop_roi(frame, car_roi)

    return donkey_img, car_img


def extract_score_rois_adaptive(frame):
    """
    Новый API: возвращает small + wide ROI.
    """
    return {
        "donkey_small": crop_roi(frame, roi_to_pixels(frame, DONKEY_ROI_SMALL)),
        "donkey_wide": crop_roi(frame, roi_to_pixels(frame, DONKEY_ROI_WIDE)),
        "car_small": crop_roi(frame, roi_to_pixels(frame, CAR_ROI_SMALL)),
        "car_wide": crop_roi(frame, roi_to_pixels(frame, CAR_ROI_WIDE)),
    }


def draw_score_rois(frame):
    """
    Рисует small и wide ROI.
    """
    rois = [
        ("DONKEY_SMALL", DONKEY_ROI_SMALL, (255, 255, 0)),
        ("DONKEY_WIDE", DONKEY_ROI_WIDE, (255, 128, 0)),
        ("CAR_SMALL", CAR_ROI_SMALL, (0, 255, 255)),
        ("CAR_WIDE", CAR_ROI_WIDE, (0, 128, 255)),
    ]

    for label, roi_rel, color in rois:
        x, y, w, h = roi_to_pixels(frame, roi_rel)

        cv.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            1,
        )

        cv.putText(
            frame,
            label,
            (x, max(15, y - 5)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv.LINE_AA,
        )


# ============================================================
# SCORE PREPROCESSING
# ============================================================

def preprocess_score_image(image):
    """
    Делает binary mask:
        цифра = белая
        фон = чёрный

    Подходит под твой score:
        - бирюзовый фон
        - тёмно-серый квадрат под цифрой
        - светло-серая цифра
    """
    if image is None or image.size == 0:
        return None

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1]

    # Бирюзовый фон цветной, saturation высокая.
    # Серый фон и цифра имеют saturation ниже.
    grayish_mask = saturation < 120

    vals = gray[grayish_mask]
    if vals.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)

    # Внутри grayish пикселей есть:
    # - тёмный серый фон
    # - светлая серая цифра
    #
    # Берём верхнюю часть яркости.
    bg_level = float(np.median(vals))
    p75 = float(np.percentile(vals, 75))

    threshold = max(bg_level + 20.0, p75, 85.0)

    mask = (grayish_mask & (gray >= threshold)).astype(np.uint8) * 255

    # Чуть склеиваем антиалиасинг цифры.
    kernel = np.ones((2, 2), dtype=np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    return mask


def crop_to_content(binary_img):
    if binary_img is None or binary_img.size == 0:
        return None

    ys, xs = np.where(binary_img > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    return binary_img[y1:y2 + 1, x1:x2 + 1]


def normalize_digit(binary_img, size=DIGIT_NORMALIZED_SIZE, pad=2):
    """
    Приводит любую цифру к одному размеру.
    """
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

    resized = cv.resize(
        cropped,
        (new_w, new_h),
        interpolation=cv.INTER_NEAREST,
    )

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)

    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


# ============================================================
# SCORE TEMPLATES
# ============================================================

def load_score_templates(score_templates_dir: str) -> dict[int, np.ndarray]:
    """
    Загружает:
        digit_0.png
        digit_1.png
        ...
        digit_9.png

    Каждый файл должен содержать одну цифру, не весь score.
    """
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


def debug_templates(templates: dict[int, np.ndarray]):
    """
    Вызови вручную после загрузки templates, если распознавание странное.
    """
    for digit, img in sorted(templates.items()):
        white_pixels = int(np.count_nonzero(img))
        print(
            f"[TEMPLATE DEBUG] digit={digit}, "
            f"shape={img.shape}, white_pixels={white_pixels}"
        )
        cv.imwrite(f"debug_template_{digit}.png", img)


# ============================================================
# SCORE DIGIT SPLIT
# ============================================================

def split_score_digits(score_roi_img, max_digits: int = 4, debug_prefix: str | None = None):
    """
    Разбивает score переменной длины:
        0
        4
        10
        41
        99
        100

    Важно:
    Эта версия НЕ склеивает соседние цифры через gap <= 2.
    Иначе 41 превращается в одну "цифру".
    """
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

    # Колонки, где есть белые пиксели цифры.
    col_has_pixels = np.any(binary > 0, axis=0)

    segments = []
    in_segment = False
    start = 0

    for x, has_pixels in enumerate(col_has_pixels):
        if has_pixels and not in_segment:
            start = x
            in_segment = True

        elif not has_pixels and in_segment:
            end = x
            segments.append((start, end))
            in_segment = False

    if in_segment:
        segments.append((start, w))

    # Убираем только совсем мелкий шум.
    segments = [
        (x1, x2)
        for x1, x2 in segments
        if (x2 - x1) >= 1
    ]

    if not segments:
        return []

    # ВАЖНО:
    # Не делаем merge по gap <= 2.
    # Иначе "41" превращается в один сегмент.
    #
    # Разрешаем merge только если сегменты реально соприкасаются.
    merged = []
    cur_start, cur_end = segments[0]

    for x1, x2 in segments[1:]:
        gap = x1 - cur_end

        if gap <= 0:
            cur_end = x2
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = x1, x2

    merged.append((cur_start, cur_end))
    segments = merged

    if len(segments) > max_digits:
        segments = segments[-max_digits:]

    digit_images = []
    debug_view = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)

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
            cv.rectangle(
                debug_view,
                (x1, 0),
                (x2, h - 1),
                (0, 255, 0),
                1,
            )

            cv.imwrite(f"{debug_prefix}_digit_{i}.png", digit_norm)

    if debug_prefix is not None:
        cv.imwrite(f"{debug_prefix}_03_segments.png", debug_view)

    return digit_images


# ============================================================
# SCORE PREDICTION
# ============================================================

def _dice_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)

    a_count = int(np.count_nonzero(a_bin))
    b_count = int(np.count_nonzero(b_bin))

    if a_count + b_count == 0:
        return 0.0

    intersection = int(np.count_nonzero((a_bin == 1) & (b_bin == 1)))

    return (2.0 * intersection) / (a_count + b_count)


def predict_score_digit(digit_img, templates: dict[int, np.ndarray]):
    """
    Предсказывает одну цифру.

    Не используем matchTemplate, потому что на маленьких binary images
    фон начинает доминировать. Dice similarity сравнивает форму цифры.
    """
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


def _predict_score_value_meta(
    score_roi_img,
    templates: dict[int, np.ndarray],
    max_digits: int = 4,
    debug_prefix: str | None = None,
):
    """
    Возвращает:
        value, confidence, digit_count
    """
    digit_images = split_score_digits(
        score_roi_img,
        max_digits=max_digits,
        debug_prefix=debug_prefix,
    )

    if not digit_images:
        return None, 0.0, 0

    digits = []
    scores = []

    for i, digit_img in enumerate(digit_images):
        pred_digit, pred_score = predict_score_digit(digit_img, templates)

        if pred_digit is None:
            return None, 0.0, 0

        digits.append(str(pred_digit))
        scores.append(float(pred_score))

    confidence = min(scores) if scores else 0.0
    value = int("".join(digits))

    return value, confidence, len(digits)


def predict_score_value(score_roi_img, templates: dict[int, np.ndarray]):
    """
    Старый API.
    """
    value, confidence, _ = _predict_score_value_meta(
        score_roi_img,
        templates,
        max_digits=4,
        debug_prefix=None,
    )

    return value, confidence


def _choose_best_score_candidate(candidates):
    """
    candidates:
        list of (value, confidence, digit_count, source_name)

    Логика:
    1. Если есть число с большим количеством цифр и нормальной уверенностью,
       выбираем его.
    2. Иначе выбираем по confidence.
    """
    valid = [
        c for c in candidates
        if c[0] is not None and c[1] >= SCORE_INTERNAL_MIN_CONF
    ]

    if not valid:
        return None, 0.0

    # Сначала предпочитаем больше цифр.
    # Это важно для ситуации:
    # small ROI прочитал "10", wide ROI прочитал "100".
    valid.sort(
        key=lambda c: (c[2], c[1]),
        reverse=True,
    )

    best_value, best_conf, best_digit_count, best_source = valid[0]

    return best_value, best_conf


def read_score_counters(
    frame,
    templates: dict[int, np.ndarray],
    debug: bool = False,
):
    rois = extract_score_rois_adaptive(frame)

    donkey_value, donkey_conf, donkey_digits = _predict_score_value_meta(
        rois["donkey_wide"],
        templates,
        max_digits=4,
        debug_prefix="debug_donkey_wide" if debug else None,
    )

    car_value, car_conf, car_digits = _predict_score_value_meta(
        rois["car_wide"],
        templates,
        max_digits=4,
        debug_prefix="debug_car_wide" if debug else None,
    )

    if debug:
        cv.imwrite("debug_donkey_wide_raw.png", rois["donkey_wide"])
        cv.imwrite("debug_car_wide_raw.png", rois["car_wide"])

        print(
            "[SCORE DEBUG] donkey_wide:",
            donkey_value,
            donkey_conf,
            "digits:",
            donkey_digits,
        )
        print(
            "[SCORE DEBUG] car_wide:",
            car_value,
            car_conf,
            "digits:",
            car_digits,
        )

    return {
        "donkey": donkey_value,
        "driver": car_value,
        "donkey_conf": donkey_conf,
        "driver_conf": car_conf,
    }


# ============================================================
# OBJECT DETECTION
# ============================================================

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

    res = cv.matchTemplate(
        frame_gray,
        template,
        cv.TM_CCOEFF_NORMED,
    )

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
        bottom_right = (
            top_left[0] + w,
            top_left[1] + h,
        )

        center = (
            top_left[0] + w // 2,
            top_left[1] + h // 2,
        )

        cv.rectangle(
            frame,
            top_left,
            bottom_right,
            color,
            2,
        )

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


# ============================================================
# STATE VECTOR
# ============================================================

def build_state(
    player_result: dict,
    donkey_result: dict,
    frame_shape: tuple,
) -> np.ndarray:
    """
    State vector size = 7:

    [
        px_n,
        py_n,
        dx_n,
        dy_n,
        rel_x,
        rel_y,
        distance
    ]

    Не меняю размер state, чтобы не ломать твою модель.
    """
    height, width = frame_shape[:2]

    def _coords(result):
        if result.get("found"):
            cx, cy = result["center"]
            return cx, cy, cx / width, cy / height

        return None, None, None, None

    px, py, px_n, py_n = _coords(player_result)
    dx, dy, dx_n, dy_n = _coords(donkey_result)

    rel_x = float(
        (dx_n - px_n)
        if (px_n is not None and dx_n is not None)
        else 0.0
    )

    rel_y = float(
        (dy_n - py_n)
        if (py_n is not None and dy_n is not None)
        else 0.0
    )

    distance = float(np.sqrt(rel_x ** 2 + rel_y ** 2))

    state = np.array(
        [
            float(px_n if px_n is not None else 0.0),
            float(py_n if py_n is not None else 0.0),
            float(dx_n if dx_n is not None else 0.0),
            float(dy_n if dy_n is not None else 0.0),
            rel_x,
            rel_y,
            distance,
        ],
        dtype=np.float32,
    )

    return state