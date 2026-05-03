import os
import time
import subprocess
from collections import deque

import cv2 as cv
import numpy as np
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
from agents.dgn_agent import DQNAgent
from agents.perform_action import perform_action

MIN_CONF = 0.35



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


def _compute_reward(
        prev_stable_driver: int,
        prev_stable_donkey: int,
        curr_stable_driver: int,
        curr_stable_donkey: int,
) -> tuple[float, bool]:
    if any(v is None for v in (prev_stable_driver, prev_stable_donkey,curr_stable_driver, curr_stable_donkey)):
        return 0.0, False

    if prev_stable_donkey < curr_stable_donkey:
        return -100.0, True

    if prev_stable_driver < curr_stable_driver:
        return 50.0 + 0.1, False

    return 0.1, False


def validate_paths():
    required = {
        "DOSBox.exe":         DOSBOX_PATH,
        "dosbox.conf":        CONF_PATH,
        "player_template.png": PLAYER_TEMPLATE_PATH,
        "donkey_template.png": DONKEY_TEMPLATE_PATH,
    }
    for name, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")




def game_step(
        region,
        templates: dict,
        prev_state: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    frame = capture_screen(region)

    player_result = detect_one(
        frame, PLAYER_TEMPLATE_PATH,
        label="player", threshold=0.80, color=(0, 255, 0),
    )
    donkey_result = detect_one(
        frame, DONKEY_TEMPLATE_PATH,
        label="donkey", threshold=0.75, color=(0, 0, 255),
    )

    state = build_state(player_result, donkey_result, frame.shape, prev_state)
    counters = read_score_counters(frame, templates)

    return state, counters




def run_episode(
        region,
        templates: dict,
        agent: DQNAgent,
        episode_idx: int = 0,
        step_interval: float = 0.0,
) -> tuple[float, list[np.ndarray]]:
    
    driver_history = deque(maxlen=2)
    donkey_history = deque(maxlen=2)

    driver_stable = None
    donkey_stable = None
    prev_stable_driver = None
    prev_stable_donkey = None
    prev_state = None

    total_reward = 0.0
    states = []
    step = 0

    while True:
        state, counters = game_step(region, templates, prev_state)
        states.append(state)

        donkey_raw = counters["donkey"] if counters["donkey_conf"] >= MIN_CONF else None
        driver_raw = counters["driver"] if counters["driver_conf"] >= MIN_CONF else None

        donkey_confirmed = _update_stable_value(donkey_history, donkey_raw)
        driver_confirmed = _update_stable_value(driver_history, driver_raw)

        if donkey_confirmed is not None:
            donkey_stable = donkey_confirmed
        if driver_confirmed is not None:
            driver_stable = driver_confirmed

        if prev_stable_donkey is None and donkey_stable is not None:
            prev_stable_donkey = donkey_stable
        if prev_stable_driver is None and driver_stable is not None:
            prev_stable_driver = driver_stable

        reward, done = _compute_reward(
            prev_stable_driver, prev_stable_donkey,
            driver_stable, donkey_stable,
        )
        total_reward += reward

        action=agent.select_action(state)
        perform_action(action)

        if prev_state is not None:
            agent.remember(prev_state,action,reward,state,done)
            agent.train_step()

        ts = time.strftime("%H:%M:%S")
        print(
            f"[{ts}] ep ={episode_idx:3d} | "
            f"donkey_stable ={donkey_stable!s:>3} "
            f"(conf={counters['donkey_conf']:.2f}) | "
            f"driver_stable ={driver_stable!s:>3} "
            f"(conf={counters['driver_conf']:.2f}) | "
            f"reward ={reward:+7.1f}  total ={total_reward:+8.1f} | ",
            f"epsilon={agent.epsilon:.3f}"
        )

        if driver_stable is not None:
            prev_stable_driver = driver_stable
        if donkey_stable is not None:
            prev_stable_donkey = donkey_stable
        prev_state = state
        step += 1

        if done:
            print(
                f"\n{'=-' * 40}\n"
                f"Episode {episode_idx} finished after {step} steps  |  "
                f"total reward = {total_reward:.1f}\n"
                f"{'=-' * 40}\n"
            )
            break

        if step_interval > 0:
            time.sleep(step_interval)

    return total_reward, states




def run_training(num_episodes: int = 500, step_interval: float = 0.0):
    validate_paths()

    score_templates_dir = os.path.join(IMAGE_TEMPLATE_DIR, "score_templates")
    templates = load_score_templates(score_templates_dir)
    if not templates:
        raise RuntimeError(
            "No score templates found. "
            f"Put digit_0.png..digit_9.png into {score_templates_dir}"
        )

    process = None
    try:
        print("Starting DOSBox…")
        process = subprocess.Popen([DOSBOX_PATH, "-conf", CONF_PATH])
        time.sleep(3)

        window = find_dosbox_window(timeout=10)
        activate_window(window)
        print(f"Window: left={window.left} top={window.top} "
              f"w={window.width} h={window.height}")

        region = get_capture_region(window)

        pyautogui.press("space")
        time.sleep(1)

        episode_rewards: list[float] = []
        agent=DQNAgent()
        for ep in range(num_episodes):
            total_reward, _ = run_episode(
                region, templates,
                agent,
                episode_idx=ep,
                step_interval=step_interval,
            )
            episode_rewards.append(total_reward)

            time.sleep(0.3)

    except KeyboardInterrupt:
        print("Training stopped by user.")
    finally:
        if process is not None:
            process.terminate()

if __name__ == "__main__":
    agent = DQNAgent()
    
    state = np.random.rand(9).astype(np.float32)
    action = agent.select_action(state)
    print(f"Action: {action}")
    
    next_state = np.random.rand(9).astype(np.float32)
    agent.remember(state, action, 0.1, next_state, False)
    print(f"Buffer size: {len(agent.replay_buffer)}")