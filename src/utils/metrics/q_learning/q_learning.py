from src.utils.metrics.base import BaseTrainingTracker

class DQNTrainingTracker(BaseTrainingTracker):
    STEP_FIELDS = [
        "state_px",
        "state_py",
        "state_dx",
        "state_dy",
        "rel_x",
        "rel_y",
        "distance",
        "player_visible",
        "donkey_visible",
        "action",
        "reward",
        "done",
        "loss",
        "epsilon",
    ]

    EPISODE_FIELDS = [
        "total_reward",
        "episode_steps",
        "lap_count",
        "crash",
        "mean_loss",
        "epsilon",
    ]

    EPISODE_SUMMARIES = {
        "reward": ("mean", "std", "min", "max"),
        "loss": ("mean", "std", "min", "max"),
    }