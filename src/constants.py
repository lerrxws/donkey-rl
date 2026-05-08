import os

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
BAD_JUMP_PENALTY = -10.0
BAD_SIDE_JUMP_PENALTY = -12.0
MISSED_DANGER_PENALTY = -8.0
GOOD_DANGER_JUMP_REWARD = 3.0

