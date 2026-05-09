from collections import deque
import time

import numpy as np

from src.config import (
    PLAYER_TEMPLATE_PATH,
    DONKEY_TEMPLATE_PATH,
    CRASH_REWARD,
    AgentMode
)
from src.detection import detect_one,build_state,read_score_counters
from src.utils.capture import capture_screen
from src.utils.counter_tracker import get_raw_counter, update_stable_value
from src.env.reward import compute_score_reward, compute_reward
from src.env.state_builder import extract_position_flags, build_simple_state
from agents.perform_action import perform_action


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

    raw_state = build_state(
        player_result,
        donkey_result,
        frame.shape
    )

    counters = read_score_counters(frame,templates)

    return raw_state, counters


def run_episode(
    region,
    templates: dict,
    agent,
    mode: AgentMode,
    episode_idx: int = 0,
    step_interval: float = 0.15,
) -> tuple[float, list[np.ndarray]]:

    driver_history = deque(maxlen=2)
    donkey_history = deque(maxlen=2)

    driver_stable = None
    donkey_stable = None

    prev_stable_driver = None
    prev_stable_donkey = None

    crash_detected = False

    total_reward = 0.0
    states: list[np.ndarray] = []

    step = 0

    print(f"[ep={episode_idx}] Waiting for player...")
    for attempt in range(50):
        raw_state, counters = game_step(region, templates)

        if raw_state[7] > 0.5:
            print(f"[ep={episode_idx}] Player found after {attempt} attempts")
            break

        time.sleep(0.2)
    else:
        print(f"[ep={episode_idx}] WARNING: player not found, starting anyway")

    while True:
        state = build_simple_state(raw_state)
        states.append(state)

        action = agent.select_action(state)

        perform_action(action)

        if step_interval > 0:
            time.sleep(step_interval)

        next_raw_state, counters = game_step(region, templates)

        donkey_raw = get_raw_counter(counters, "donkey")
        driver_raw = get_raw_counter(counters, "driver")

        donkey_confirmed = update_stable_value(donkey_history, donkey_raw)
        driver_confirmed = update_stable_value(driver_history, driver_raw)

        if donkey_confirmed is not None:
            donkey_stable = donkey_confirmed

        if driver_confirmed is not None:
            driver_stable = driver_confirmed

        if prev_stable_donkey is None and donkey_stable is not None:
            prev_stable_donkey = donkey_stable

        if prev_stable_driver is None and driver_stable is not None:
            prev_stable_driver = driver_stable


        base_reward, done = compute_score_reward(
            prev_stable_driver,
            prev_stable_donkey,
            driver_stable,
            donkey_stable,
        )


        reward = compute_reward(
            base_reward=base_reward,
            done=done,
            raw_state=raw_state,
            action=action,
        )

        total_reward += reward

        next_state = build_simple_state(next_raw_state)

        agent.remember(state,action,reward,next_state,done)

        if mode == AgentMode.DQN:
            agent.train_step()


        print(
            f"[{time.strftime("%H:%M:%S")}] ep={episode_idx:4d} | "
            f"step={step:4d} | "
            f"donkey={donkey_stable!s:>4} driver={driver_stable!s:>4} | "
            f"crash={crash_detected} | "
            f"r={reward:+7.1f} total={total_reward:+8.1f}"
        )

        if driver_stable is not None:
            prev_stable_driver = driver_stable

        if donkey_stable is not None:
            prev_stable_donkey = donkey_stable

        raw_state = next_raw_state
        step += 1

        if done:
            print(
                f"\n{'=-' * 40}\n"
                f"Episode {episode_idx} finished after {step} steps | "
                f"total reward = {total_reward:.1f}\n"
                f"{'=-' * 40}\n"
            )
            break

    return total_reward, states