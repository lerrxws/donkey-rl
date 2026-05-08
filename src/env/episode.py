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
        frame.shape,
    )

    counters = read_score_counters(
        frame,
        templates,
    )

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

    prev_raw_donkey = None
    prev_raw_driver = None

    crash_detected = False

    total_reward = 0.0
    states: list[np.ndarray] = []

    action_counts = {
        0: 0,
        1: 0,
    }

    bad_jump_count = 0
    side_jump_count = 0
    missed_jump_count = 0
    good_jump_count = 0

    step = 0

    episode_losses: list[float] = []

    print(f"[ep={episode_idx}] Waiting for player...")
    for attempt in range(50):
        raw_state, counters = game_step(region, templates)

        if raw_state[7] > 0.5:
            print(f"[ep={episode_idx}] Player found after {attempt} attempts")
            break

        time.sleep(0.2)
    else:
        print(f"[ep={episode_idx}] WARNING: player not found, starting anyway")

    initial_donkey_raw = get_raw_counter(counters, "donkey")
    if initial_donkey_raw is not None:
        prev_raw_donkey = initial_donkey_raw

    initial_driver_raw = get_raw_counter(counters, "driver")
    if initial_driver_raw is not None:
        prev_raw_driver = initial_driver_raw

    while True:
        state = build_simple_state(raw_state)
        states.append(state)

        action = agent.select_action(state)

        perform_action(action)
        action_counts[action] += 1

        if step_interval > 0:
            time.sleep(step_interval)

        next_raw_state, counters = game_step(region, templates)

        donkey_raw = get_raw_counter(counters, "donkey")
        driver_raw = get_raw_counter(counters, "driver")

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

        next_player_visible = next_raw_state[7] > 0.5

        base_reward, done = compute_score_reward(
            prev_stable_driver,
            prev_stable_donkey,
            driver_stable,
            donkey_stable,
        )

        current_player_visible = raw_state[7] > 0.5

        player_was_visible = current_player_visible
        player_now_gone = not next_player_visible
        were_close = raw_state[6] < 0.30

        if (
            player_was_visible
            and player_now_gone
            and were_close
            and not raw_lap
            and not crash_detected
        ):
            crash_detected = True
            base_reward = CRASH_REWARD
            done = True

        if raw_crash and not crash_detected:
            crash_detected = True
            base_reward = CRASH_REWARD
            done = True
        elif raw_crash and crash_detected:
            raw_crash = False

        reward = compute_reward(
            base_reward=base_reward,
            done=done,
            raw_state=raw_state,
            action=action,
        )

        flags = extract_position_flags(raw_state)

        danger = flags["danger"]
        donkey_visible = flags["donkey_visible"]
        same_line = flags["same_line"]

        side_jump = action == 1 and donkey_visible and not same_line
        bad_jump = action == 1 and not danger
        missed_jump = action == 0 and danger
        good_jump = action == 1 and danger

        if mode == AgentMode.ACTOR_CRITIC:
            side_jump_count += int(side_jump)
            bad_jump_count += int(bad_jump)
            missed_jump_count += int(missed_jump)
            good_jump_count += int(good_jump)

        total_reward += reward

        next_state = build_simple_state(next_raw_state)

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done,
        )

        loss_value = None
        if mode == AgentMode.DQN:
            loss_value = agent.train_step()

        if loss_value is not None:
            episode_losses.append(loss_value)

        ts = time.strftime("%H:%M:%S")

        # probs = agent.last_probs
        # prob_str = (
        #     f"probs=[no_jump:{probs[0]:.3f},jump:{probs[1]:.3f}]"
        #     if probs is not None
        #     else ""
        # )

        # print(
        #     f"[{ts}] ep={episode_idx:4d} | "
        #     f"step={step:4d} | "
        #     f"action={executed_action} | "
        #     f"donkey_stable={donkey_stable!s:>4} | "
        #     f"driver_stable={driver_stable!s:>4} | "
        #     f"reward={reward:+7.1f} "
        #     f"total={total_reward:+8.1f} | "
        #     # f"epsilon={agent.epsilon:.3f}"
        # )

        if mode == AgentMode.ACTOR_CRITIC:
            print(
                f"[{ts}] ep={episode_idx:4d} | "
                f"step={step:4d} | "
                f"a={action} | "
                f"car_line={int(flags['car_line'])} | "
                f"donkey_line={int(flags['donkey_line'])} | "
                f"same_line={int(flags['same_line'])} | "
                f"danger={int(flags['danger'])} | "
                f"rel_x={flags['rel_x']:+.3f} | "
                f"rel_y={flags['rel_y']:+.3f} | "
                f"side_jump={int(side_jump)} | "
                f"bad_jump={int(bad_jump)} | "
                f"missed={int(missed_jump)} | "
                f"good_jump={int(good_jump)} | "
                f"vis=[p:{int(current_player_visible)},d:{int(flags['donkey_visible'])}] | "
                f"donkey={donkey_stable!s:>4} driver={driver_stable!s:>4} | "
                f"crash={crash_detected} | "
                f"r={reward:+7.1f} total={total_reward:+8.1f}"
            )
        else:
            print(
                f"[{ts}] ep={episode_idx:4d} | "
                f"step={step:4d} | "
                f"action={action} | "
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
            avg_loss = float(np.mean(episode_losses)) if episode_losses else None
            loss_text = f" | avg_loss={avg_loss:.4f}" if avg_loss is not None else ""
            print(
                f"\n{'=-' * 40}\n"
                f"Episode {episode_idx} finished after {step} steps | "
                f"total reward = {total_reward:.1f}{loss_text}\n"
                f"actions = {action_counts}\n"
                f"side_jumps = {side_jump_count} | "
                f"bad_jumps = {bad_jump_count} | "
                f"missed_jumps = {missed_jump_count} | "
                f"good_jumps = {good_jump_count}\n"
                f"{'=-' * 40}\n"
            )
            break

    return total_reward, states