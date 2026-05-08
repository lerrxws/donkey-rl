import numpy as np

from src.config import (
    LINE_SPLIT_X,
    LANE_THRESHOLD,
    DANGER_Y_MIN,
    DANGER_Y_MAX
)


def x_to_line(x: float, visible: bool) -> float:
    if not visible:
        return -1.0

    if x < LINE_SPLIT_X:
        return 0.0

    return 1.0


def extract_position_flags(raw_state: np.ndarray) -> dict:
    player_x = float(raw_state[0])
    donkey_x = float(raw_state[2])

    rel_x = float(raw_state[4])
    rel_y = float(raw_state[5])

    player_visible = float(raw_state[7]) > 0.5
    donkey_visible = float(raw_state[8]) > 0.5

    car_line = x_to_line(player_x, player_visible)
    donkey_line = x_to_line(donkey_x, donkey_visible)

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


def build_simple_state(raw_state: np.ndarray) -> np.ndarray:
    flags = extract_position_flags(raw_state)

    return np.array(
        [
            float(flags["car_line"]),
            float(flags["donkey_line"]),
            float(flags["danger"]),
        ],
        dtype=np.float32,
    )