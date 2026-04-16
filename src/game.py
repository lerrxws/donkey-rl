import os
import time
import subprocess

import cv2 as cv
import pyautogui

from src.constants import (
    DOSBOX_PATH,
    CONF_PATH,
    IMAGE_DEBUG_DIR,
    PLAYER_TEMPLATE_PATH,
    DONKEY_TEMPLATE_PATH,
    DEBUG_RAW_PATH,
    DEBUG_RESULT_PATH,
)
from src.window import find_dosbox_window, activate_window, get_capture_region
from src.capture import capture_screen
from src.detection import detect_one, build_state


def validate_paths():
    required = {
        "DOSBox.exe": DOSBOX_PATH,
        "dosbox.conf": CONF_PATH,
        "player_template.png": PLAYER_TEMPLATE_PATH,
        "donkey_template.png": DONKEY_TEMPLATE_PATH,
    }
    for name, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} не знайдено: {path}")


def run_detection_once():
    validate_paths()
    process = None
    try:
        print("Starting DOSBox...")
        process = subprocess.Popen([DOSBOX_PATH, "-conf", CONF_PATH])
        time.sleep(3)

        window = find_dosbox_window(timeout=10)
        activate_window(window)

        print(f"Window found: left={window.left}, top={window.top}, "
              f"width={window.width}, height={window.height}")

        region = get_capture_region(window)
        print("Capture region:", region)

        print("Sending SPACE...")
        pyautogui.press("space")
        time.sleep(2)

        frame = capture_screen(region)
        os.makedirs(IMAGE_DEBUG_DIR, exist_ok=True)

        if not cv.imwrite(DEBUG_RAW_PATH, frame):
            raise RuntimeError(f"Failed to save the raw frame: {DEBUG_RAW_PATH}")
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

        if not cv.imwrite(DEBUG_RESULT_PATH, frame):
            raise RuntimeError(f"Failed to save the result frame: {DEBUG_RESULT_PATH}")
        print("Saved result frame to:", DEBUG_RESULT_PATH)

        cv.imshow("Detection result", frame)
        cv.waitKey(0)
        cv.destroyAllWindows()

        return state

    finally:
        if process is not None:
            process.terminate()