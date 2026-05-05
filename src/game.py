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
from src.utils.rewarding_system import (
    compute_score_reward,
    danger_reward,
    unnecessary_action_penalty,
    post_action_danger_penalty,
    CRASH_REWARD
    )


from src.seed_init import set_seed
from agents.dgn_agent import DQNAgent
from agents.perform_action import perform_action


MIN_CONF = 0.35

SPACE_COOLDOWN_STEPS = 2
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



def _looks_like_collision(state: np.ndarray) -> bool:
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
        return False

    same_lane = abs(rel_x) < 0.08

    collision_y = -0.30 < rel_y < 0.12

    return same_lane and collision_y

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

    prev_raw_driver = None
    prev_raw_donkey = None

    lost_player_frames = 0
    # space_cooldown = 0
    crash_detected = False  

    total_reward = 0.0
    states: list[np.ndarray] = []

    step = 0

    episode_losses = []
    avg_loss = None

    print(f"[ep={episode_idx}] Waiting for player...")
    for attempt in range(50): 
        state, counters = game_step(region, templates)
        if state[7] > 0.5:  
            print(f"[ep={episode_idx}] Player found after {attempt} attempts")
            break
        time.sleep(0.2)
    else:
        print(f"[ep={episode_idx}] WARNING: player not found, starting anyway")

    initial_donkey_raw = _get_raw_counter(counters, "donkey")
    if initial_donkey_raw is not None:
        prev_raw_donkey = initial_donkey_raw

    initial_driver_raw = _get_raw_counter(counters, "driver")
    if initial_driver_raw is not None:
        prev_raw_driver = initial_driver_raw

    while True:
        
        states.append(state)

        selected_action = agent.select_action(state)
        executed_action = selected_action

        # if space_cooldown > 0 and executed_action == 1:
        #     executed_action = 0

        perform_action(executed_action)

        # if executed_action == 1:
        #     space_cooldown = SPACE_COOLDOWN_STEPS
        # else:
        #     space_cooldown = max(0, space_cooldown - 1)

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


        raw_lap = False

        if prev_raw_driver is not None and driver_raw is not None:
            if driver_raw > prev_raw_driver:
                raw_lap = True

        if driver_raw is not None:
            if prev_raw_driver is None or driver_raw >= prev_raw_driver:
                prev_raw_driver = driver_raw

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

        reward, done, score_event =  compute_score_reward(
            prev_stable_driver,
            prev_stable_donkey,
            driver_stable,
            donkey_stable,
        )

        current_player_visible = state[7] > 0.5
        current_donkey_visible = state[8] > 0.5

        player_was_visible = current_player_visible
        player_now_gone = not next_player_visible

        if (
            player_was_visible
            and player_now_gone
            and _looks_like_collision(state)
            and not raw_lap
            and not crash_detected
        ):
            crash_detected = True
            reward = CRASH_REWARD
            done = True

        if raw_crash and not crash_detected:
            crash_detected = True
            reward = CRASH_REWARD
            done = True
        elif raw_crash and crash_detected:
            raw_crash = False

        if not done and score_event is None:
            if not current_player_visible:
                reward = min(reward, -5.0)

            elif not current_donkey_visible:
                reward = min(reward, 0.0)

            reward += danger_reward(state, executed_action)
            reward += unnecessary_action_penalty(state, executed_action)
            # reward += post_action_danger_penalty(next_state)

            if lost_player_frames >= LOST_PLAYER_LIMIT:
                reward = min(reward, -10.0)

        total_reward += reward

        agent.remember(
            state,
            executed_action,
            reward,
            next_state,
            done,
        )

        loss_value=agent.train_step()
        if loss_value is not None:
            episode_losses.append(loss_value)

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
            avg_loss = float(np.mean(episode_losses))
            loss_text = f"{avg_loss:.4f}" if avg_loss is not None else "NA"
            print(
                f"\n{'=-' * 40}\n"
                f"Episode {episode_idx} finished after {step} steps | "
                f"total reward = {total_reward:.1f} | "
                f"avg_loss = {loss_text}\n"
                f"{'=-' * 40}\n"
            )
            break

    return total_reward, states


def run_training(
    num_episodes: int = 20000,
    step_interval: float = 0.15,
):
    start_time=time.perf_counter()
    set_seed(42)
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

        region = get_capture_region(window)

        agent = DQNAgent()

        pyautogui.press("space")
        time.sleep(1)

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

    except KeyboardInterrupt:
        print("Training stopped by user.")

    finally:
        end_time=time.perf_counter()
        elapsed=end_time-start_time
        print(
            f"Training was running for {elapsed:.1f} seconds "
            f"({elapsed / 60:.2f} minutes)"
        )
        if process is not None:
            process.terminate()


if __name__ == "__main__":
    run_training(
        num_episodes=20000,
        step_interval=0.15,
    )