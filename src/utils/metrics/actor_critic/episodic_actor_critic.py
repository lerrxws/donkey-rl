from src.utils.metrics.actor_critic.actor_critic import ActorCriticTracker


class EpisodicActorCriticTracker(ActorCriticTracker):
    EPISODE_FIELDS = ActorCriticTracker.EPISODE_FIELDS + [
        "actor_loss",
        "critic_loss",
        "entropy_mean",
        "value",
        "target",
        "advantage",
        "mean_value",
        "mean_target",
        "mean_advantage",
        "batch_size",
        "total_raw_reward",
    ]

    EPISODE_SUMMARIES = {
        "entropy_mean": ("mean", "std"),
        "value": ("mean", "min", "max"),
        "prob_no_jump": ("mean",),
        "prob_jump": ("mean",),
        "selected_action_prob": ("mean",),
    }
