import time

import pygetwindow as gw


def is_main_dosbox_window(title: str) -> bool:
    if not title:
        return False

    t = " ".join(title.split())
    t_lower = t.lower()

    if "status window" in t_lower:
        return False

    return (
        "dosbox" in t_lower
        and "program:" in t_lower
        and "donkey" in t_lower
    )


def find_dosbox_window(timeout=10):
    start = time.time()

    while time.time() - start < timeout:
        all_titles = gw.getAllTitles()
        print("All window titles:")
        for t in all_titles:
            if t.strip():
                print("  ", repr(t))

        candidates = [
            w
            for w in gw.getAllWindows()
            if is_main_dosbox_window(w.title or "") and w.width > 0 and w.height > 0
        ]

        if candidates:
            candidates.sort(key=lambda w: w.width * w.height, reverse=True)
            return candidates[0]

        time.sleep(0.5)

    raise RuntimeError("Unable to find the main DOSBox window from the 'Program: DONKEY' menu")


def activate_window(window):
    try:
        window.activate()
        time.sleep(0.5)
    except Exception:
        pass


def get_capture_region(window) -> dict:
    border_left = 8
    border_right = 8
    title_bar = 31
    border_bottom = 8

    left = window.left + border_left
    top = window.top + title_bar
    width = window.width - border_left - border_right
    height = window.height - title_bar - border_bottom

    if width <= 0 or height <= 0:
        raise RuntimeError("Incorrect capture area dimensions")

    return {"left": left, "top": top, "width": width, "height": height}