import os
import time
import subprocess

import cv2 as cv
import numpy as np
import mss
import pyautogui
import pygetwindow as gw


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONF_PATH = os.path.join(BASE_DIR, "dosbox.conf")
DOSBOX_PATH = r"C:\Program Files (x86)\DOSBox-0.74-3\DOSBox.exe"

PLAYER_TEMPLATE_PATH = os.path.join(BASE_DIR, "player_template.png")
DONKEY_TEMPLATE_PATH = os.path.join(BASE_DIR, "donkey_template.png")

DEBUG_RAW_PATH = os.path.join(BASE_DIR, "debug_raw.png")
DEBUG_RESULT_PATH = os.path.join(BASE_DIR, "debug_result.png")


def find_dosbox_window(timeout=10):
    start = time.time()

    while time.time() - start < timeout:
        all_titles = gw.getAllTitles()
        print("All window titles:")
        for t in all_titles:
            if t.strip():
                print("  ", repr(t))

        candidates = []
        for w in gw.getAllWindows():
            title = w.title or ""
            if is_main_dosbox_window(title) and w.width > 0 and w.height > 0:
                candidates.append(w)

        if candidates:
            candidates.sort(key=lambda w: w.width * w.height, reverse=True)
            return candidates[0]

        time.sleep(0.5)

    raise RuntimeError("Не вдалося знайти головне вікно DOSBox з Program: DONKEY")

def is_main_dosbox_window(title: str) -> bool:
    if not title:
        return False

    t = " ".join(title.split())  # прибирає зайві пробіли
    t_lower = t.lower()

    if "status window" in t_lower:
        return False

    return (
        "dosbox" in t_lower and
        "program:" in t_lower and
        "donkey" in t_lower
    )

def activate_window(window):
    try:
        window.activate()
        time.sleep(0.5)
    except Exception:
        pass


def get_capture_region(window):
    # Обрізаємо рамку та title bar приблизно.
    # Для Windows/DOSBox це зазвичай працює достатньо добре.
    border_left = 8
    border_right = 8
    title_bar = 31
    border_bottom = 8

    left = window.left + border_left
    top = window.top + title_bar
    width = window.width - border_left - border_right
    height = window.height - title_bar - border_bottom

    if width <= 0 or height <= 0:
        raise RuntimeError("Невірні розміри області захоплення")

    return {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
    }


def capture_screen(region):
    with mss.mss() as sct:
        screenshot = np.array(sct.grab(region))
    frame = cv.cvtColor(screenshot, cv.COLOR_BGRA2BGR)
    return frame


def detect_one(frame, template_path, label, threshold=0.8, color=(0, 255, 0)):
    template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Не знайдено template: {template_path}")

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]

    res = cv.matchTemplate(frame_gray, template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

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


def build_state(player_result, donkey_result, frame_shape):
    height, width = frame_shape[:2]

    state = {
        "player_found": int(player_result["found"]),
        "donkey_found": int(donkey_result["found"]),
    }

    if player_result["found"]:
        px, py = player_result["center"]
        state["player_x"] = px
        state["player_y"] = py
        state["player_x_norm"] = px / width
        state["player_y_norm"] = py / height
    else:
        state["player_x"] = None
        state["player_y"] = None
        state["player_x_norm"] = None
        state["player_y_norm"] = None

    if donkey_result["found"]:
        dx, dy = donkey_result["center"]
        state["donkey_x"] = dx
        state["donkey_y"] = dy
        state["donkey_x_norm"] = dx / width
        state["donkey_y_norm"] = dy / height
    else:
        state["donkey_x"] = None
        state["donkey_y"] = None
        state["donkey_x_norm"] = None
        state["donkey_y_norm"] = None

    if player_result["found"] and donkey_result["found"]:
        state["dx"] = donkey_result["center"][0] - player_result["center"][0]
        state["dy"] = donkey_result["center"][1] - player_result["center"][1]
    else:
        state["dx"] = None
        state["dy"] = None

    return state


def main():
    print("BASE_DIR:", BASE_DIR)
    print("CONF_PATH:", CONF_PATH)
    print("DOSBOX_PATH:", DOSBOX_PATH)
    print("PLAYER_TEMPLATE_PATH:", PLAYER_TEMPLATE_PATH)
    print("DONKEY_TEMPLATE_PATH:", DONKEY_TEMPLATE_PATH)

    if not os.path.exists(DOSBOX_PATH):
        raise FileNotFoundError("DOSBox.exe не знайдено")
    if not os.path.exists(CONF_PATH):
        raise FileNotFoundError("dosbox.conf не знайдено")
    if not os.path.exists(PLAYER_TEMPLATE_PATH):
        raise FileNotFoundError("player_template.png не знайдено")
    if not os.path.exists(DONKEY_TEMPLATE_PATH):
        raise FileNotFoundError("donkey_template.png не знайдено")

    process = None

    try:
        print("Starting DOSBox...")
        process = subprocess.Popen([DOSBOX_PATH, "-conf", CONF_PATH])

        time.sleep(3)

        window = find_dosbox_window(timeout=10)
        activate_window(window)

        print("Window found:")
        print(" left =", window.left)
        print(" top =", window.top)
        print(" width =", window.width)
        print(" height =", window.height)

        region = get_capture_region(window)
        print("Capture region:", region)

        print("Sending SPACE...")
        pyautogui.press("space")
        time.sleep(2)

        frame = capture_screen(region)
        cv.imwrite(DEBUG_RAW_PATH, frame)
        print("Saved raw frame to:", DEBUG_RAW_PATH)

        player_result = detect_one(
            frame,
            PLAYER_TEMPLATE_PATH,
            label="player",
            threshold=0.80,
            color=(0, 255, 0),
        )

        donkey_result = detect_one(
            frame,
            DONKEY_TEMPLATE_PATH,
            label="donkey",
            threshold=0.75,
            color=(0, 0, 255),
        )

        print("Player result:", player_result)
        print("Donkey result:", donkey_result)

        state = build_state(player_result, donkey_result, frame.shape)
        print("State:", state)

        cv.imwrite(DEBUG_RESULT_PATH, frame)
        print("Saved result frame to:", DEBUG_RESULT_PATH)

        cv.imshow("Detection result", frame)
        cv.waitKey(0)
        cv.destroyAllWindows()

    finally:
        if process is not None:
            process.terminate()


if __name__ == "__main__":
    main()