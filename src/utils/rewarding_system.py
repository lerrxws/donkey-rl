import numpy as np

STEP_REWARD = 1.0
LAP_REWARD = 100.0
CRASH_REWARD = -100.0


def compute_score_reward(
    prev_stable_driver: int | None,
    prev_stable_donkey: int | None,
    curr_stable_driver: int | None,
    curr_stable_donkey: int | None,
) -> tuple[float, bool, str | None]:

    if any(v is None for v in (
        prev_stable_driver,
        prev_stable_donkey,
        curr_stable_driver,
        curr_stable_donkey,
    )):
        return 0.0, False, None

    if curr_stable_donkey > prev_stable_donkey:
        return CRASH_REWARD, True, "crash"

    if curr_stable_driver > prev_stable_driver:
        return LAP_REWARD + STEP_REWARD, False, "lap"

    return STEP_REWARD, False, None


def danger_reward(state: np.ndarray, action: int) -> float:
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

    danger_y = -0.65 < rel_y < -0.18

    if same_lane and danger_y:
        if action == 1:
            return +12.0
        return -12.0

    return 0.0

def unnecessary_action_penalty(state: np.ndarray, action: int) -> float:
    if action != 1:
        return 0.0

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

    if player_visible < 0.5:
        return -2.0

    if donkey_visible < 0.5:
        return -2.0

    same_lane = abs(rel_x) < 0.06
    danger_y = -0.65 < rel_y < -0.18

    if same_lane and danger_y:
        return 0.0

    return -4.0

def post_action_danger_penalty(next_state: np.ndarray) -> float:
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
    ) = next_state

    if player_visible < 0.5 or donkey_visible < 0.5:
        return 0.0

    same_lane = abs(rel_x) < 0.06
    danger_y = -0.65 < rel_y < -0.18

    if same_lane and danger_y:
        return -8.0

    return 0.0