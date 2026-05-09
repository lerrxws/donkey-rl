import cv2 as cv
import numpy as np
import mss


def capture_screen(region: dict):
    with mss.mss() as sct:
        screenshot = np.array(sct.grab(region))
    frame = cv.cvtColor(screenshot, cv.COLOR_BGRA2BGR)
    return frame