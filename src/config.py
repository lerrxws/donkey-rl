import os
from enum import Enum

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
IMAGE_DIR = os.path.join(PROJECT_ROOT, "data")

IMAGE_TEMPLATE_DIR = os.path.join(IMAGE_DIR, "templates")
IMAGE_SCORE_TEMPLATE_DIR = os.path.join(IMAGE_TEMPLATE_DIR, "score_templates")
IMAGE_DEBUG_DIR = os.path.join(IMAGE_DIR, "debug")

CONF_PATH = os.path.join(PROJECT_ROOT, "dosbox.conf")
DOSBOX_PATH = r"C:\Program Files (x86)\DOSBox-0.74-3\DOSBox.exe"

PLAYER_TEMPLATE_PATH = os.path.join(IMAGE_TEMPLATE_DIR, "player_template.png")
DONKEY_TEMPLATE_PATH = os.path.join(IMAGE_TEMPLATE_DIR, "donkey_template.png")

DEBUG_RAW_PATH = os.path.join(IMAGE_DEBUG_DIR, "debug_raw.png")
DEBUG_RESULT_PATH = os.path.join(IMAGE_DEBUG_DIR, "debug_result.png")


MIN_CONF = 0.35


STEP_REWARD = 1.0
LAP_REWARD = 100.0
CRASH_REWARD = -100.0
BAD_JUMP_PENALTY = -10.0
BAD_SIDE_JUMP_PENALTY = -12.0
MISSED_DANGER_PENALTY = -8.0
GOOD_DANGER_JUMP_REWARD = 3.0


STATE_SIZE = 3


LINE_SPLIT_X = 0.50

LANE_THRESHOLD = 0.06

DANGER_Y_MIN = -0.65
DANGER_Y_MAX = -0.18


class AgentMode(str, Enum):
    ACTOR_CRITIC = "actor_critic"
    DQN = "dqn"
    DOUBLE_DQN = "double_dqn"

