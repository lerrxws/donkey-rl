import os
import time
import subprocess
from collections import deque

import cv2 as cv
import pyautogui

from src.constants import (
    DOSBOX_PATH,
    CONF_PATH,
    IMAGE_TEMPLATE_DIR,
    IMAGE_DEBUG_DIR,
    PLAYER_TEMPLATE_PATH,
    DONKEY_TEMPLATE_PATH,
    DEBUG_RAW_PATH,
    DEBUG_RESULT_PATH,
)
from src.window import find_dosbox_window, activate_window, get_capture_region
from src.capture import capture_screen
from src.detection import (
    detect_one,
    build_state,
    draw_score_rois,
    extract_score_rois,
    load_score_templates,
    read_score_counters,
)


def _update_stable_value(history: deque, candidate):
    if candidate is None:
        return None

    history.append(candidate)
    if len(history) < history.maxlen:
        return None

    first = history[0]
    if all(v == first for v in history):
        history.clear()
        return first

    return None



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

        donkey_roi_img, car_roi_img = extract_score_rois(frame)
        draw_score_rois(frame)

        donkey_roi_path = os.path.join(IMAGE_DEBUG_DIR, "donkey_roi.png")
        car_roi_path = os.path.join(IMAGE_DEBUG_DIR, "car_roi.png")

        if not cv.imwrite(donkey_roi_path, donkey_roi_img):
            raise RuntimeError(f"Failed to save donkey ROI image: {donkey_roi_path}")
        if not cv.imwrite(car_roi_path, car_roi_img):
            raise RuntimeError(f"Failed to save car ROI image: {car_roi_path}")
        print("Saved donkey ROI to:", donkey_roi_path)
        print("Saved car ROI to:", car_roi_path)

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


def capture_donkey_score_templates(interval_sec: float = 2.0):
    validate_paths()

    output_dir = os.path.join(IMAGE_TEMPLATE_DIR, "donkey_score_samples")
    os.makedirs(output_dir, exist_ok=True)

    process = None
    sample_idx = 0
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
        time.sleep(1)

        print(f"Capturing Donkey score ROI every {interval_sec:.1f}s. Press Ctrl+C to stop.")
        while True:
            frame = capture_screen(region)
            donkey_roi_img, _ = extract_score_rois(frame)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(
                output_dir,
                f"donkey_score_{timestamp}_{sample_idx:04d}.png",
            )

            if not cv.imwrite(out_path, donkey_roi_img):
                raise RuntimeError(f"Failed to save donkey score ROI image: {out_path}")

            print("Saved donkey score ROI to:", out_path)
            sample_idx += 1
            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        if process is not None:
            process.terminate()


def print_score_counters_every_second(interval_sec: float = 1.0):
    validate_paths()

    score_templates_dir = os.path.join(IMAGE_TEMPLATE_DIR, "score_templates")
    templates = load_score_templates(score_templates_dir)
    if not templates:
        raise RuntimeError(
            "No score templates found. Put digit_0.png..digit_9.png or donkey_score_*_0000..0009.png "
            f"into {score_templates_dir}"
        )

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
        time.sleep(1)

        print(f"Reading score counters every {interval_sec:.1f}s. Press Ctrl+C to stop.")
        
        
        while True:
            driver_score_history = deque(maxlen=3)
            donkey_score_history = deque(maxlen=3)

            driver_stable = None
            donkey_stable = None

            min_conf=0.35

            score = 0
            prev_stable_driver = None 
            prev_stable_donkey = None 

            episode_done = False

            while not episode_done:
                frame = capture_screen(region)
                counters = read_score_counters(frame, templates)
                timestamp = time.strftime("%H:%M:%S")

                donkey_raw = counters["donkey"] if counters["donkey_conf"] >= min_conf else None
                driver_raw = counters["driver"] if counters["driver_conf"] >= min_conf else None

                donkey_confirmed = _update_stable_value(donkey_score_history, donkey_raw)
                driver_confirmed = _update_stable_value(driver_score_history, driver_raw)

                if donkey_confirmed is not None:
                    donkey_stable = donkey_confirmed
                if driver_confirmed is not None:
                    driver_stable = driver_confirmed

                if prev_stable_driver is None and driver_stable is not None:
                    prev_stable_driver = driver_stable
                if prev_stable_donkey is None and donkey_stable is not None:
                    prev_stable_donkey = donkey_stable

                score, episode_done = _compute_reward_penalties(
                    prev_stable_driver=prev_stable_driver,
                    prev_stable_donkey=prev_stable_donkey,
                    curr_stable_driver=driver_stable,
                    curr_stable_donkey=donkey_stable,
                    score=score,
                )

                if driver_stable is not None:
                    prev_stable_driver = driver_stable
                if donkey_stable is not None:
                    prev_stable_donkey = donkey_stable

                
                
                donkey_raw_str = "?" if donkey_raw is None else str(donkey_raw)
                driver_raw_str = "?" if driver_raw is None else str(driver_raw)
                donkey_stable_str = "?" if donkey_stable is None else str(donkey_stable)
                driver_stable_str = "?" if driver_stable is None else str(driver_stable)

                print(
                    f"[{timestamp}] Donkey score={donkey_raw_str} stable={donkey_stable_str} "
                    f"(conf={counters['donkey_conf']:.3f}) | "
                    f"Driver score={driver_raw_str} stable={driver_stable_str} "
                    f"(conf={counters['driver_conf']:.3f}) | "
                    f"score={score}"
                )
                if episode_done:
                    print(
                        f"[{timestamp}] Episode ended. Total score: {score}\n",
                        60*"=-"
                    )
                    break
                time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        if process is not None:
            process.terminate()

def _compute_reward_penalties(prev_stable_driver, prev_stable_donkey, 
                               curr_stable_driver, curr_stable_donkey, score):
    done = False

    if any(v is None for v in (prev_stable_driver, prev_stable_donkey, 
                                curr_stable_driver, curr_stable_donkey)):
        return score, done

    if prev_stable_donkey < curr_stable_donkey:
        score -= 100
        done = True
        return score, done

    if prev_stable_driver < curr_stable_driver:
        score += 50

    score += 0.1

    return score, done