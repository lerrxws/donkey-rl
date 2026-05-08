from utils.metrics.base import BaseTrainingTracker


class ActorCriticTracker(BaseTrainingTracker):
    STEP_FIELDS = [
        "reward",
        "total_reward",
        "action",
        "done",
        "danger",
        "same_line",
        "good_jump",
        "bad_jump",
        "missed_jump",
        "side_jump",
        "crash",
        "value",
        "entropy_mean",
        "prob_no_jump",
        "prob_jump",
        "selected_action_prob",
    ]

    EPISODE_FIELDS = [
        "total_reward",
        "episode_steps",
    ]

    # Aggregated from per-step metrics into one row in episodes.csv.
    EPISODE_SUMMARIES: dict[str, tuple[str, ...]] = {}
