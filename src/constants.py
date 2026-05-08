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
