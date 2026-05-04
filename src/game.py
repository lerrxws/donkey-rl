import os
import time
import subprocess
from collections import deque

import numpy as np
import pyautogui

from src.constants import (
    DOSBOX_PATH,
    CONF_PATH,
    IMAGE_TEMPLATE_DIR,
    PLAYER_TEMPLATE_PATH,
    DONKEY_TEMPLATE_PATH,
)

from src.window import find_dosbox_window, activate_window, get_capture_region
from src.capture import capture_screen

from src.detection import (
    detect_one,
    build_state,
    load_score_templates,
    read_score_counters,
)

from agents.dgn_agent import DQNAgent
from agents.perform_action import perform_action


MIN_CONF = 0.35

STEP_REWARD = 1.0
LAP_REWARD = 100.0
CRASH_REWARD = -100.0

SPACE_COOLDOWN_STEPS = 3
LOST_PLAYER_LIMIT = 3


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


def _get_raw_counter(counters: dict, name: str):
    value = counters.get(name)
    conf = counters.get(f"{name}_conf", 0.0)

    if value is None:
        return None

    if conf < MIN_CONF:
        return None

    return int(value)


def _compute_score_reward(
    prev_stable_driver: int | None,
    prev_stable_donkey: int | None,
    curr_stable_driver: int | None,
    curr_stable_donkey: int | None,
) -> tuple[float, bool]:
    if any(v is None for v in (
        prev_stable_driver,
        prev_stable_donkey,
        curr_stable_driver,
        curr_stable_donkey,
    )):
        return 0.0, False

    if curr_stable_donkey > prev_stable_donkey:
        return CRASH_REWARD, True

    if curr_stable_driver > prev_stable_driver:
        return LAP_REWARD + STEP_REWARD, False

    return STEP_REWARD, False


def _danger_reward(state: np.ndarray, action: int) -> float:
    """
    Дополнительный shaping reward.

    Это не teacher. Агент всё равно сам выбирает action.
    Мы просто даём reward раньше, когда ситуация опасная.
    """
    (
        px_n,
        py_n,
        dx_n,
        dy_n,
        rel_x,
        rel_y,
        distance,
        player_visible,
        donkey_visible,
    ) = state

    if player_visible < 0.5 or donkey_visible < 0.5:
        return 0.0

    same_lane = abs(rel_x) < 0.06
    danger_y = -0.45 < rel_y < -0.12

    if same_lane and danger_y:
        if action == 1:
            return +8.0
        return -8.0

    return 0.0


def validate_paths():
    required = {
        "DOSBox.exe": DOSBOX_PATH,
        "dosbox.conf": CONF_PATH,
        "player_template.png": PLAYER_TEMPLATE_PATH,
        "donkey_template.png": DONKEY_TEMPLATE_PATH,
    }

    for name, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")


def game_step(region, templates: dict) -> tuple[np.ndarray, dict]:
    frame = capture_screen(region)

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

    state = build_state(
        player_result,
        donkey_result,
        frame.shape,
    )

    counters = read_score_counters(
        frame,
        templates,
    )

    return state, counters


def run_episode(
    region,
    templates: dict,
    agent: DQNAgent,
    episode_idx: int = 0,
    step_interval: float = 0.15,
) -> tuple[float, list[np.ndarray]]:

    driver_history = deque(maxlen=2)
    donkey_history = deque(maxlen=2)

    driver_stable = None
    donkey_stable = None

    prev_stable_driver = None
    prev_stable_donkey = None

    prev_raw_donkey = None

    lost_player_frames = 0
    space_cooldown = 0

    total_reward = 0.0
    states: list[np.ndarray] = []

    action_counts = {
        0: 0,
        1: 0,
    }

    step = 0

    state, counters = game_step(region, templates)

    initial_donkey_raw = _get_raw_counter(counters, "donkey")
    if initial_donkey_raw is not None:
        prev_raw_donkey = initial_donkey_raw

    while True:
        states.append(state)

        selected_action = agent.select_action(state)
        executed_action = selected_action

        if space_cooldown > 0 and executed_action == 1:
            executed_action = 0

        perform_action(executed_action)

        action_counts[executed_action] += 1

        if executed_action == 1:
            space_cooldown = SPACE_COOLDOWN_STEPS
        else:
            space_cooldown = max(0, space_cooldown - 1)

        if step_interval > 0:
            time.sleep(step_interval)

        next_state, counters = game_step(region, templates)

        donkey_raw = _get_raw_counter(counters, "donkey")
        driver_raw = _get_raw_counter(counters, "driver")

        raw_crash = False

        if prev_raw_donkey is not None and donkey_raw is not None:
            if donkey_raw > prev_raw_donkey:
                raw_crash = True

        if donkey_raw is not None:
            if prev_raw_donkey is None or donkey_raw >= prev_raw_donkey:
                prev_raw_donkey = donkey_raw

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

        next_player_visible = next_state[7] > 0.5

        if not next_player_visible:
            lost_player_frames += 1
        else:
            lost_player_frames = 0

        reward, done = _compute_score_reward(
            prev_stable_driver,
            prev_stable_donkey,
            driver_stable,
            donkey_stable,
        )

        if not done:
            reward += _danger_reward(state, executed_action)

        if raw_crash:
            reward = CRASH_REWARD
            done = True

        if lost_player_frames >= LOST_PLAYER_LIMIT and not done:
            reward = min(reward, -10.0)

        total_reward += reward

        agent.remember(
            state,
            executed_action,
            reward,
            next_state,
            done,
        )

        agent.train_step()

        print(
            "state=",
            np.round(state, 3),
            "selected_action=",
            selected_action,
            "executed_action=",
            executed_action,
        )

        ts = time.strftime("%H:%M:%S")

        print(
            f"[{ts}] ep={episode_idx:4d} | "
            f"step={step:4d} | "
            f"action={executed_action} | "
            f"donkey_stable={donkey_stable!s:>4} | "
            f"driver_stable={driver_stable!s:>4} | "
            f"lost_player={lost_player_frames} | "
            f"raw_crash={raw_crash} | "
            f"reward={reward:+7.1f} "
            f"total={total_reward:+8.1f} | "
            f"epsilon={agent.epsilon:.3f}"
        )

        if driver_stable is not None:
            prev_stable_driver = driver_stable

        if donkey_stable is not None:
            prev_stable_donkey = donkey_stable

        state = next_state
        step += 1

        if done:
            print(
                f"\n{'=-' * 40}\n"
                f"Episode {episode_idx} finished after {step} steps | "
                f"total reward = {total_reward:.1f}\n"
                f"actions = {action_counts}\n"
                f"{'=-' * 40}\n"
            )
            break

    return total_reward, states


def run_training(
    num_episodes: int = 20000,
    step_interval: float = 0.15,
):
    validate_paths()

    score_templates_dir = os.path.join(
        IMAGE_TEMPLATE_DIR,
        "score_templates",
    )

    templates = load_score_templates(score_templates_dir)

    if not templates:
        raise RuntimeError(
            "No score templates found. "
            f"Put digit_0.png..digit_9.png into {score_templates_dir}"
        )

    process = None

    try:
        print("Starting DOSBox...")
        process = subprocess.Popen([DOSBOX_PATH, "-conf", CONF_PATH])

        time.sleep(3)

        window = find_dosbox_window(timeout=10)
        activate_window(window)

        print(
            f"Window: left={window.left} top={window.top} "
            f"w={window.width} h={window.height}"
        )

        region = get_capture_region(window)

        pyautogui.press("space")
        time.sleep(1)

        agent = DQNAgent()

        episode_rewards: list[float] = []

        for ep in range(num_episodes):
            total_reward, _ = run_episode(
                region=region,
                templates=templates,
                agent=agent,
                episode_idx=ep,
                step_interval=step_interval,
            )

            episode_rewards.append(total_reward)

            avg_last_10 = float(np.mean(episode_rewards[-10:]))

            print(
                f"[TRAIN] episode={ep} "
                f"reward={total_reward:.1f} "
                f"avg_last_10={avg_last_10:.1f} "
                f"epsilon={agent.epsilon:.3f}"
            )

            time.sleep(0.3)

    except KeyboardInterrupt:
        print("Training stopped by user.")

    finally:
        if process is not None:
            process.terminate()


if __name__ == "__main__":
    run_training(
        num_episodes=20000,
        step_interval=0.15,
    )