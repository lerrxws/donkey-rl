from .tracker import BaseTrainingTracker


class EpisodicActorCriticTracker(BaseTrainingTracker):
    STEP_METRIC_FIELDS = [
        "entropy",
        "value",
        "prob_no_jump",
        "prob_jump",
        "selected_action_prob",
    ]

    METRIC_SUMMARIES = {
        "entropy": ("mean", "std"),
        "value": ("mean",),
        "prob_no_jump": ("mean",),
        "prob_jump": ("mean",),
        "selected_action_prob": ("mean",),

        # updates after episode
        "actor_loss": ("mean",),
        "critic_loss": ("mean",),
        "return": ("mean", "std", "min", "max"),
        "advantage": ("mean", "abs_mean", "std", "min", "max"),
    }
