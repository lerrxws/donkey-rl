from .tracker import BaseTrainingTracker


class OneStepActorCriticTracker(BaseTrainingTracker):
    STEP_METRIC_FIELDS = [
        "actor_loss",
        "critic_loss",
        "td_error",
        "entropy",
        "value",
        "target",
        "scaled_reward",
        "prob_no_jump",
        "prob_jump",
        "selected_action_prob",
    ]

    METRIC_SUMMARIES = {
        "actor_loss": ("mean", "std"),
        "critic_loss": ("mean", "std"),
        "td_error": ("mean", "abs_mean", "std", "min", "max"),
        "entropy": ("mean", "std"),
        "value": ("mean",),
        "target": ("mean",),
        "scaled_reward": ("mean",),
        "prob_no_jump": ("mean",),
        "prob_jump": ("mean",),
        "selected_action_prob": ("mean",),
    }
