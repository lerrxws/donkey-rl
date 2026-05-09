import numpy as np


def build_state(
    player_result: dict,
    donkey_result: dict,
    frame_shape: tuple,
) -> np.ndarray:
    height, width = frame_shape[:2]

    def _normalized_coords(result: dict) -> tuple[float | None, float | None]:
        if result.get("found"):
            cx, cy = result["center"]
            return cx / width, cy / height
        return None, None

    px_n, py_n = _normalized_coords(player_result)
    dx_n, dy_n = _normalized_coords(donkey_result)

    player_visible = px_n is not None
    donkey_visible = dx_n is not None

    if player_visible and donkey_visible:
        rel_x = float(dx_n - px_n)
        rel_y = float(dy_n - py_n)
        distance = float(np.sqrt(rel_x ** 2 + rel_y ** 2))
    else:
        rel_x, rel_y, distance = 0.0, 0.0, 1.0

    return np.array(
        [
            float(px_n if px_n is not None else 0.0),
            float(py_n if py_n is not None else 0.0),
            float(dx_n if dx_n is not None else 0.0),
            float(dy_n if dy_n is not None else 0.0),
            rel_x,
            rel_y,
            distance,
            float(player_visible),
            float(donkey_visible),
        ],
        dtype=np.float32,
    )