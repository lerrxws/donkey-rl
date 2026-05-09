from collections import deque

from src.config import MIN_CONF

def update_stable_value(history: deque, candidate):
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


def get_raw_counter(counters: dict, name: str):
    value = counters.get(name)
    conf = counters.get(f"{name}_conf", 0.0)

    if value is None:
        return None

    if conf < MIN_CONF:
        return None

    return int(value)