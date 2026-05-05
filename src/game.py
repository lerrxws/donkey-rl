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
from src.seed_init import set_seed
from agents.actor_critic_agent import MonteCarloActorCriticAgent
from agents.perform_action import perform_action


MIN_CONF = 0.35

STEP_REWARD = 1.0
LAP_REWARD = 100.0
CRASH_REWARD = -100.0

LOST_PLAYER_LIMIT = 3

# State:
#   0: car_line
#   1: donkey_line
#   2: danger
#
# car_line/donkey_line:
#   -1 = object not visible
#    0 = left lane
#    1 = right lane
#
# danger:
#    1 = donkey is visible, in the same lane, and in jump timing zone
#    0 = not dangerous right now
STATE_SIZE = 3

# If normalized x < 0.5 => left lane, else right lane.
# If your capture crop is asymmetric, tune this value using logs.
LINE_SPLIT_X = 0.50

# Extra tolerance: if car and donkey x are close enough, treat them as same line
# even if LINE_SPLIT_X puts them on different sides near the boundary.
LANE_THRESHOLD = 0.06

# Y zone where jumping is useful.
# These values are from your older danger_y logic.
DANGER_Y_MIN = -0.65
DANGER_Y_MAX = -0.18

# Reward shaping.
BAD_JUMP_PENALTY = -10.0       # jumped when danger=0
BAD_SIDE_JUMP_PENALTY = -12.0  # jumped when donkey is visible but in another line
MISSED_DANGER_PENALTY = -8.0   # did not jump when danger=1
GOOD_DANGER_JUMP_REWARD = 3.0  # jumped when danger=1


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


def _x_to_line(x: float, visible: bool) -> float:
    """
    Converts normalized X coordinate into lane id.

    Returns:
        -1.0 if object is not visible
         0.0 for left lane
         1.0 for right lane
    """
    if not visible:
        return -1.0

    if x < LINE_SPLIT_X:
        return 0.0

    return 1.0


def _extract_position_flags(raw_state: np.ndarray) -> dict:
    """
    raw_state comes from build_state().

    Expected indices:
        raw_state[0] = player x normalized
        raw_state[1] = player y normalized
        raw_state[2] = donkey x normalized
        raw_state[3] = donkey y normalized
        raw_state[4] = rel_x
        raw_state[5] = rel_y
        raw_state[7] = player_visible
        raw_state[8] = donkey_visible
    """
    player_x = float(raw_state[0])
    donkey_x = float(raw_state[2])

    rel_x = float(raw_state[4])
    rel_y = float(raw_state[5])

    player_visible = float(raw_state[7]) > 0.5
    donkey_visible = float(raw_state[8]) > 0.5

    car_line = _x_to_line(player_x, player_visible)
    donkey_line = _x_to_line(donkey_x, donkey_visible)

    same_line_by_id = (
        player_visible
        and donkey_visible
        and car_line >= 0.0
        and donkey_line >= 0.0
        and int(car_line) == int(donkey_line)
    )

    same_line_by_distance = (
        player_visible
        and donkey_visible
        and abs(rel_x) < LANE_THRESHOLD
    )

    same_line = same_line_by_id or same_line_by_distance

    danger = (
        donkey_visible
        and same_line
        and DANGER_Y_MIN < rel_y < DANGER_Y_MAX
    )

    return {
        "player_x": player_x,
        "donkey_x": donkey_x,
        "rel_x": rel_x,
        "rel_y": rel_y,
        "player_visible": player_visible,
        "donkey_visible": donkey_visible,
        "car_line": car_line,
        "donkey_line": donkey_line,
        "same_line": same_line,
        "danger": danger,
    }


def _build_simple_state(raw_state: np.ndarray) -> np.ndarray:
    """
    State [3]:
        0: car_line
        1: donkey_line
        2: danger

    This is better than one binary jump_zone because the agent can distinguish:
        - donkey in same line;
        - donkey in another line;
        - actual danger timing.
    """
    flags = _extract_position_flags(raw_state)

    return np.array(
        [
            float(flags["car_line"]),
            float(flags["donkey_line"]),
            float(flags["danger"]),
        ],
        dtype=np.float32,
    )


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


def _compute_reward(
    base_reward: float,
    done: bool,
    player_visible: bool,
    raw_state: np.ndarray,
    action: int,
) -> float:
    """
    Reward:
        +1     survival step
        +101   driver score increased
        -100   crash
        -12    jumped when donkey is visible but in another line
        -10    jumped when danger=0
        -8     did not jump when danger=1
        +3     jumped when danger=1
    """
    if done:
        return base_reward

    reward = base_reward

    flags = _extract_position_flags(raw_state)

    donkey_visible = flags["donkey_visible"]
    same_line = flags["same_line"]
    danger = flags["danger"]

    jumped = action == 1

    if jumped and donkey_visible and not same_line:
        reward += BAD_SIDE_JUMP_PENALTY

    elif jumped and not danger:
        reward += BAD_JUMP_PENALTY

    elif not jumped and danger:
        reward += MISSED_DANGER_PENALTY

    elif jumped and danger:
        reward += GOOD_DANGER_JUMP_REWARD

    # Keep this disabled unless player detection is stable.
    # if not player_visible:
    #     reward = min(reward, -5.0)

    return reward


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


def _format_episode_metrics(agent) -> str:
    """Formats metrics after finish_episode()."""
    m = agent.last_metrics
    if m is None:
        return ""

    probs = agent.last_probs
    prob_str = ""

    if probs is not None and len(probs) >= 2:
        prob_str = f" | probs=[no_jump:{probs[0]:.3f},jump:{probs[1]:.3f}]"

    return (
        f" | loss={m['loss']:+.4f}"
        f" | actor={m['actor_loss']:+.4f}"
        f" | critic={m['critic_loss']:+.4f}"
        f" | adv={m['mean_advantage']:+.4f}"
        f" | V={m['mean_value']:+.4f}"
        f" | H={m['entropy']:.4f}"
        f"{prob_str}"
    )


def run_episode(
    region,
    templates: dict,
    agent: MonteCarloActorCriticAgent,
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

    print(f"[ep={episode_idx}] Waiting for player...")

    for attempt in range(50):
        raw_state, counters = game_step(region, templates)

        if raw_state[7] > 0.5:
            print(f"[ep={episode_idx}] Player found after {attempt} attempts")
            break

        time.sleep(0.2)

    else:
        print(f"[ep={episode_idx}] WARNING: player not found, starting anyway")

    initial_donkey_raw = _get_raw_counter(counters, "donkey")

    if initial_donkey_raw is not None:
        prev_raw_donkey = initial_donkey_raw

    while True:
        state = _build_simple_state(raw_state)
        states.append(state)

        action = agent.select_action(state)

        perform_action(action)
        action_counts[action] += 1

        if step_interval > 0:
            time.sleep(step_interval)

        next_raw_state, counters = game_step(region, templates)

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

        next_player_visible = next_raw_state[7] > 0.5

        if not next_player_visible:
            lost_player_frames += 1
        else:
            lost_player_frames = 0

        base_reward, done = _compute_score_reward(
            prev_stable_driver,
            prev_stable_donkey,
            driver_stable,
            donkey_stable,
        )

        current_player_visible = raw_state[7] > 0.5

        player_was_visible = current_player_visible
        player_now_gone = not next_player_visible
        were_close = raw_state[6] < 0.30

        if player_was_visible and player_now_gone and were_close and not crash_detected:
            crash_detected = True
            base_reward = CRASH_REWARD
            done = True

        if raw_crash and not crash_detected:
            crash_detected = True
            base_reward = CRASH_REWARD
            done = True

        elif raw_crash and crash_detected:
            raw_crash = False

        reward = _compute_reward(
            base_reward=base_reward,
            done=done,
            player_visible=current_player_visible,
            raw_state=raw_state,
            action=action,
        )

        flags = _extract_position_flags(raw_state)

        danger = flags["danger"]
        donkey_visible = flags["donkey_visible"]
        same_line = flags["same_line"]

        side_jump = action == 1 and donkey_visible and not same_line
        bad_jump = action == 1 and not danger
        missed_jump = action == 0 and danger
        good_jump = action == 1 and danger

        side_jump_count += int(side_jump)
        bad_jump_count += int(bad_jump)
        missed_jump_count += int(missed_jump)
        good_jump_count += int(good_jump)

        total_reward += reward

        next_state = _build_simple_state(next_raw_state)

        # MC: accumulate transitions; training happens in finish_episode().
        agent.remember(
            state,
            action,
            reward,
            next_state,
            done,
        )

        ts = time.strftime("%H:%M:%S")

        probs = agent.last_probs
        prob_str = (
            f"probs=[no_jump:{probs[0]:.3f},jump:{probs[1]:.3f}]"
            if probs is not None
            else ""
        )

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
            f"r={reward:+7.1f} total={total_reward:+8.1f} | "
            f"{prob_str}"
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
                f"actions = {action_counts}\n"
                f"side_jumps = {side_jump_count} | "
                f"bad_jumps = {bad_jump_count} | "
                f"missed_jumps = {missed_jump_count} | "
                f"good_jumps = {good_jump_count}\n"
                f"{'=-' * 40}\n"
            )
            break

    return total_reward, states


def run_training(
    num_episodes: int = 20000,
    step_interval: float = 0.15,
):
    set_seed(122)
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
    agent = None

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

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

        agent = MonteCarloActorCriticAgent(
            state_size=STATE_SIZE,
            hidden_layers=[64, 64],
            gamma=0.97,
            lr=0.0003,
            entropy_coef=0.001,
            normalize_returns=True,
        )

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

            # MC training after each episode.
            agent.finish_episode()

            if hasattr(agent, "save") and (ep + 1) % 50 == 0:
                agent.save(os.path.join(checkpoint_dir, f"agent1_ep_{ep + 1}.pt"))

            avg_last_10 = float(np.mean(episode_rewards[-10:]))

            print(
                f"[TRAIN] episode={ep} "
                f"reward={total_reward:.1f} "
                f"avg_last_10={avg_last_10:.1f}"
                f"{_format_episode_metrics(agent)}"
            )

    except KeyboardInterrupt:
        print("Training stopped by user.")

    finally:
        if agent is not None and hasattr(agent, "save"):
            agent.save(os.path.join(checkpoint_dir, "agent1_last.pt"))

        if process is not None:
            process.terminate()


if __name__ == "__main__":
    run_training(
        num_episodes=20000,
        step_interval=0.15,
    )