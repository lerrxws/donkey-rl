import os
from enum import Enum

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
IMAGE_DIR = os.path.join(PROJECT_ROOT, "data")

IMAGE_TEMPLATE_DIR = os.path.join(IMAGE_DIR, "templates")
IMAGE_SCORE_TEMPLATE_DIR = os.path.join(IMAGE_TEMPLATE_DIR, "score_templates")
IMAGE_DEBUG_DIR = os.path.join(IMAGE_DIR, "debug")

RUNS_DIR = os.path.join(IMAGE_DIR, "runs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
GRAPH_DIR_NAME = "graphs"

ONE_STEP_ACTOR_CRITIC_RUN_NAME = "one_step_actor_critic"
EPISODIC_ACTOR_CRITIC_RUN_NAME = "episodic_actor_critic"
Q_LEARNING_RUN_NAME= "q_learning"
DOUBLE_Q_LEARNING_RUN_NAME= "double_q_learning"

CONF_PATH = os.path.join(PROJECT_ROOT, "dosbox.conf")
DOSBOX_PATH = r"C:\Program Files (x86)\DOSBox-0.74-3\DOSBox.exe"

PLAYER_TEMPLATE_PATH = os.path.join(IMAGE_TEMPLATE_DIR, "player_template.png")
DONKEY_TEMPLATE_PATH = os.path.join(IMAGE_TEMPLATE_DIR, "donkey_template.png")

DEBUG_RAW_PATH = os.path.join(IMAGE_DEBUG_DIR, "debug_raw.png")
DEBUG_RESULT_PATH = os.path.join(IMAGE_DEBUG_DIR, "debug_result.png")

# detection
DONKEY_ROI_SMALL = (0.118, 0.179, 0.079, 0.051)
DONKEY_ROI_WIDE  = (0.118, 0.179, 0.130, 0.051)

CAR_ROI_SMALL = (0.753, 0.179, 0.079, 0.051)
CAR_ROI_WIDE  = (0.753, 0.179, 0.130, 0.051)

DIGIT_NORMALIZED_SIZE = (24, 16)
SATURATION_THRESHOLD = 120
MIN_BRIGHTNESS_OFFSET = 20.0
MIN_BRIGHTNESS_FLOOR = 85.0
DIGIT_PAD = 2

SEGMENT_MIN_WIDTH  = 1
SEGMENT_MERGE_GAP  = 0

TEMPLATE_MATCH_THRESHOLD = 0.8

MIN_CONF = 0.35


STEP_REWARD = 1.0
LAP_REWARD = 100.0
CRASH_REWARD = -100.0
UNNECESSARY_JUMP_PENALTY= -10.0
BAD_SIDE_JUMP_PENALTY = -12.0
MISSED_DANGER_PENALTY = -8.0
GOOD_DANGER_JUMP_REWARD = 3.0


STATE_SIZE = 3
ACTION_SIZE = 2
HIDDEN_LAYERS_SIZE= [64, 64]
NUMBER_OF_SEED=64
MAX_EPISODE_STEPS=500

LINE_SPLIT_X = 0.50

LANE_THRESHOLD = 0.06

DANGER_Y_MIN = -0.65
DANGER_Y_MAX = -0.18


class AgentMode(str, Enum):
    ACTOR_CRITIC = "actor_critic"
    ONE_STEP_ACTOR_CRITIC = "one_step_actor_critic"
    EPISODIC_ACTOR_CRITIC = "episodic_actor_critic"
    DQN = "dqn"
    DOUBLE_DQN = "double_dqn"

