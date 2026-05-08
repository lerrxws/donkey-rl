from .actor_critic import ActorCriticTracker


class OneStepActorCriticTracker(ActorCriticTracker):
    STEP_FIELDS = ActorCriticTracker.STEP_FIELDS + [
        "actor_loss",
        "critic_loss",
        "td_error",
        "advantage",
        "next_value",
        "target",
        "scaled_reward",
    ]

    EPISODE_SUMMARIES = {
        "actor_loss": ("mean", "std", "min", "max"),
        "critic_loss": ("mean", "std", "min", "max"),
        "td_error": ("mean", "abs_mean", "std", "min", "max"),
        "advantage": ("mean", "abs_mean", "std", "min", "max"),
        "entropy_mean": ("mean", "std"),
        "value": ("mean", "min", "max"),
        "next_value": ("mean", "min", "max"),
        "target": ("mean", "min", "max"),
    }
