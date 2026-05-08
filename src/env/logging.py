from src.config import AgentMode

def format_episode_metrics(mode: AgentMode, agent) -> str:
    if mode == AgentMode.DQN:
        epsilon = getattr(agent, "epsilon", None)
        if epsilon is None:
            return ""
        return f" | epsilon={epsilon:.3f}"

    m = getattr(agent, "last_metrics", None)
    if m is None:
        return ""



    return (
        f" | loss={m['loss']:+.4f}"
        f" | actor={m['actor_loss']:+.4f}"
        f" | critic={m['critic_loss']:+.4f}"
        f" | adv={m['mean_advantage']:+.4f}"
        f" | V={m['mean_value']:+.4f}"
        f" | H={m['entropy']:.4f}"
    )