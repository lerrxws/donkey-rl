from src.utils.metrics.actor_critic.actor_critic import ActorCriticTracker


class EpisodicActorCriticTracker(ActorCriticTracker):
    EPISODE_SUMMARIES = {
        "entropy_mean": ("mean", "std"),
        "value": ("mean", "min", "max"),
        "prob_no_jump": ("mean",),
        "prob_jump": ("mean",),
        "selected_action_prob": ("mean",),
    }
