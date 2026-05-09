import numpy as np

from src.config import(
    CRASH_REWARD,
    LAP_REWARD,
    STEP_REWARD,
    BAD_SIDE_JUMP_PENALTY,
    UNNECESSARY_JUMP_PENALTY,
    MISSED_DANGER_PENALTY,
    GOOD_DANGER_JUMP_REWARD
)
from src.env.state_builder import extract_position_flags

def compute_score_reward(
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


def compute_reward(
    base_reward: float,
    done: bool,
    raw_state: np.ndarray,
    action: int
) -> float:
    if done:
        return base_reward

    reward = base_reward

    flags = extract_position_flags(raw_state)

    donkey_visible = flags["donkey_visible"]
    same_line = flags["same_line"]
    danger = flags["danger"]

    jumped = action == 1

    if jumped and donkey_visible and not same_line:
        reward += BAD_SIDE_JUMP_PENALTY

    elif jumped and not danger:
        reward += UNNECESSARY_JUMP_PENALTY

    elif not jumped and danger:
        reward += MISSED_DANGER_PENALTY

    elif jumped and danger:
        reward += GOOD_DANGER_JUMP_REWARD

    return reward